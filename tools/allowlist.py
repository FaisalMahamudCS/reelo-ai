"""
tools/allowlist.py
──────────────────
Permission layer: enforces which tools each user_type can call.
PII redaction logic for scrubbing sensitive data before LLM calls.

Design decisions:
  • Allowlist is a simple dict — O(1) lookup, zero overhead per request.
  • PII redaction runs on BOTH user queries AND tool results before they
    enter the LLM prompt. Two layers: regex-based text scrubbing + field-level
    key filtering on tool result dicts.
  • In production: integrate with Odoo's res.groups or your IAM provider.
  • Audit trail: log the pre-redaction hash (not content) so you can verify
    redaction occurred without storing PII in logs.
"""
import re
from typing import Any, Dict, Set


# ── Tool allowlists per user_type ─────────────────────────────────
# Tradeoff: hardcoded for speed. In production, load from config/DB
# so ops can update without code deploy.

TOOL_ALLOWLIST: Dict[str, Set[str]] = {
    "internal_sales": {
        "hot_picks",
        "compliance_filter",
        "stock_by_warehouse",
        "kb_search",
        # internal_sales cannot onboard vendors
    },
    "portal_vendor": {
        "vendor_validate",
        "kb_search",
        # vendors cannot see sales/stock data
    },
    "portal_customer": {
        "kb_search",
        # customers can only search public KB
    },
}


def check_tool_allowed(user_type: str, tool_name: str) -> bool:
    """Check if user_type is permitted to call tool_name."""
    allowed = TOOL_ALLOWLIST.get(user_type, set())
    return tool_name in allowed


def assert_tool_allowed(user_type: str, tool_name: str) -> None:
    """
    Raise PermissionError if tool is not in user's allowlist.
    Called BEFORE every tool invocation — fail-closed security.
    """
    if not check_tool_allowed(user_type, tool_name):
        raise PermissionError(
            f"User type '{user_type}' is not allowed to call tool '{tool_name}'. "
            f"Allowed tools: {sorted(TOOL_ALLOWLIST.get(user_type, set()))}"
        )


# ── PII Redaction ─────────────────────────────────────────────────
# Applied before ANY data enters an LLM prompt.
# Order matters: SSN before phone (SSN is more specific pattern).

PII_PATTERNS = [
    # SSN: 123-45-6789
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    # Email: user@domain.com
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL_REDACTED]"),
    # Phone: 123-456-7890, (123) 456-7890, 1234567890
    (re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE_REDACTED]"),
    # Credit card: 16-digit sequences
    (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CARD_REDACTED]"),
]


def redact_pii(text: str) -> str:
    """
    Scrub PII from text before it enters an LLM prompt.
    Applied to: user queries + tool result summaries.

    Production note: store original (pre-redaction) text in a separate
    encrypted audit log for compliance investigations.
    """
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_tool_result(result: dict) -> dict:
    """
    Deep-scrub a tool result dict before injecting into LLM context.
    Two levels:
      1. Remove known PII field keys entirely
      2. Redact PII patterns in string values
    """
    PII_KEYS = {"email", "phone", "ssn", "address", "tax_id", "dob", "social_security"}

    cleaned = {}
    for k, v in result.items():
        if k.lower() in PII_KEYS:
            continue
        if isinstance(v, str):
            cleaned[k] = redact_pii(v)
        elif isinstance(v, dict):
            cleaned[k] = redact_tool_result(v)
        elif isinstance(v, list):
            cleaned[k] = [
                redact_tool_result(item) if isinstance(item, dict)
                else redact_pii(item) if isinstance(item, str)
                else item
                for item in v
            ]
        else:
            cleaned[k] = v
    return cleaned
