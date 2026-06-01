"""
observability.py
────────────────
Structured JSON logging for every request.
Records: request_id, user_type, intent, tools_called + args,
         latency_ms per tool, total_latency_ms, prompt_tokens_estimate.

Design decisions:
  • Uses Python's logging module with a custom JSON formatter — compatible
    with any log aggregator (Datadog, CloudWatch, ELK, Langfuse).
  • Separate logger for traces vs application logs.
  • Token estimation: ~4 chars/token (Llama-3 BPE averages 3.5–4.5).
  • In production: ship traces to Langfuse for LLM observability, and
    structured logs to your SIEM for audit trails.
"""
import json
import logging
import sys
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


# ── Configure structured JSON logger ──────────────────────────────

class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line — machine-parseable."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
        return json.dumps(log_obj, default=str)


def setup_logging() -> None:
    """
    Configure root and application loggers.
    Call once at startup from main.py.
    """
    # JSON handler for structured output
    json_handler = logging.StreamHandler(sys.stdout)
    json_handler.setFormatter(JSONFormatter())

    # Application logger
    app_logger = logging.getLogger("reelo")
    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(json_handler)
    app_logger.propagate = False

    # Trace logger (separate so traces can be routed independently)
    trace_logger = logging.getLogger("reelo.trace")
    trace_logger.setLevel(logging.INFO)
    trace_logger.addHandler(json_handler)
    trace_logger.propagate = False

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


# ── Tool call record ──────────────────────────────────────────────

@dataclass
class ToolCall:
    tool_name: str
    args: Dict[str, Any]
    result_summary: str
    latency_ms: float
    cache_hit: bool = False


# ── Per-request trace ─────────────────────────────────────────────

@dataclass
class RequestTrace:
    """
    Accumulates observability data for a single /chat request.
    Created at request start, emitted at request end.

    Fields match the spec exactly:
      request_id, user_type, intent, tools_called + args,
      latency_ms per tool, total_latency_ms, prompt_tokens_estimate.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_type: str = ""
    intent: str = ""
    raw_query: str = ""
    tools_called: List[ToolCall] = field(default_factory=list)
    total_latency_ms: float = 0.0
    prompt_tokens_estimate: int = 0
    status: str = "ok"
    error: Optional[str] = None
    _start: float = field(default_factory=time.time, repr=False)

    def add_tool_call(
        self,
        tool_name: str,
        args: Dict,
        result_summary: str,
        latency_ms: float,
        cache_hit: bool = False,
    ) -> None:
        self.tools_called.append(
            ToolCall(tool_name, args, result_summary, latency_ms, cache_hit)
        )

    def finish(self, status: str = "ok", error: Optional[str] = None) -> None:
        """Finalize trace and emit structured log."""
        self.total_latency_ms = round((time.time() - self._start) * 1000, 2)
        self.status = status
        self.error = error
        self._emit()

    def _emit(self) -> None:
        """Emit trace as structured JSON log."""
        record = {
            "request_id": self.request_id,
            "user_type": self.user_type,
            "intent": self.intent,
            "query_preview": self.raw_query[:80],
            "tools_called": [asdict(t) for t in self.tools_called],
            "total_latency_ms": self.total_latency_ms,
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
            "status": self.status,
            "error": self.error,
        }
        logger = logging.getLogger("reelo.trace")
        logger.info("request_trace", extra={"extra_data": record})


def estimate_tokens(text: str) -> int:
    """
    Rough estimate: ~4 chars per token for Llama-3 BPE tokenizer.
    Good enough for budget tracking. For exact counts, use tiktoken.
    """
    return max(1, len(text) // 4)
