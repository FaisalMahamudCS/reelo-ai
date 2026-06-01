"""
main.py
───────
FastAPI entry point.
POST /chat — main chat endpoint
GET /health — health check with Redis status

Thin HTTP layer — all business logic lives in the orchestrator.
Startup: loads seed data + connects Redis.
Shutdown: closes Redis connection pool.
"""
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from data_loader import load_seed_data
from chains.orchestrator import get_graph
from state.session import get_or_create
from observability import RequestTrace, setup_logging
import redis_client

# Load .env before anything else
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load seed data into memory, connect Redis, setup logging.
    Shutdown: close Redis connection pool.
    """
    setup_logging()
    load_seed_data("data/seed_data.json")
    print("[STARTUP] seed_data.json loaded and validated via Pydantic")

    redis_ok = redis_client.init_redis()
    print(f"[STARTUP] Redis: {'connected' if redis_ok else 'unavailable (running without cache)'}")

    yield

    redis_client.close_redis()
    print("[SHUTDOWN] Redis connection closed")


app = FastAPI(
    title="Reelo AI — Product Chat Service",
    description="Production-oriented AI chat service with deterministic tools, "
                "compliance gating, Redis caching, and LLM formatting via Groq.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request/Response Models ───────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., description="User's natural language query", min_length=1)
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for follow-up context. Auto-generated if not provided.",
    )
    user_type: str = Field(
        default="internal_sales",
        description="internal_sales | portal_vendor | portal_customer",
    )


class ChatResponse(BaseModel):
    session_id: str
    request_id: str
    intent: str
    response: str
    basket: Optional[list] = None
    error: Optional[str] = None


# ── POST /chat ────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Classifies intent, executes tool chain,
    formats response via LLM. All in a single request cycle.
    """
    # Validate user_type
    valid_user_types = {"internal_sales", "portal_vendor", "portal_customer"}
    if req.user_type not in valid_user_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid user_type. Must be one of: {sorted(valid_user_types)}",
        )

    # Session management
    session_id = req.session_id or str(uuid.uuid4())
    session = get_or_create(session_id)

    # Observability trace
    trace = RequestTrace()
    trace.user_type = req.user_type
    trace.raw_query = req.query

    # Run LangGraph
    graph = get_graph()
    result = graph.invoke({
        "query": req.query,
        "user_type": req.user_type,
        "session": session,
        "trace": trace,
        "intent": "",
        "tool_results": {},
        "final_response": "",
        "error": None,
    })

    trace.finish(
        status="error" if result.get("error") else "ok",
        error=result.get("error"),
    )

    return ChatResponse(
        session_id=session_id,
        request_id=trace.request_id,
        intent=result["intent"],
        response=result["final_response"],
        basket=session.basket if session.basket else None,
        error=result.get("error"),
    )


# ── GET /health ───────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint with Redis connectivity status."""
    return {
        "status": "ok",
        "service": "reelo-ai-chat",
        "redis": "connected" if redis_client.is_available() else "unavailable",
    }
