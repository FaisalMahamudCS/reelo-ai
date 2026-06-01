"""
state/session.py
────────────────
Dual-layer session state: Redis (primary) + in-memory dict (L1 fallback).

Design tradeoffs:
  • Redis-backed: sessions survive server restarts, support horizontal scaling,
    and auto-expire via TTL (1 hour) — no manual cleanup.
  • In-memory fallback: if Redis is down, the system still works for the
    current process lifetime. We write-through to both layers.
  • Basket is part of session state so "add 2 of the first one" works
    across follow-up turns within the same session.
  • Production: add session versioning (optimistic locking) to prevent
    race conditions in multi-worker deployments.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import logging

import redis_client

logger = logging.getLogger("reelo.session")


@dataclass
class SessionState:
    session_id: str
    last_intent: Optional[str] = None
    last_state: Optional[str] = None          # e.g. "CA"
    last_budget: Optional[float] = None
    last_product_ids: List[int] = field(default_factory=list)
    basket: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        return cls(
            session_id=data["session_id"],
            last_intent=data.get("last_intent"),
            last_state=data.get("last_state"),
            last_budget=data.get("last_budget"),
            last_product_ids=data.get("last_product_ids", []),
            basket=data.get("basket", []),
        )


# ── In-memory L1 store (fallback if Redis unavailable) ────────────
_SESSIONS: Dict[str, SessionState] = {}


def get_or_create(session_id: str) -> SessionState:
    """
    Retrieve session: try Redis first, then in-memory, then create new.
    Write-through: newly created sessions are persisted to both layers.
    """
    # L1: check in-memory cache
    if session_id in _SESSIONS:
        return _SESSIONS[session_id]

    # L2: check Redis
    redis_data = redis_client.get_session_data(session_id)
    if redis_data:
        session = SessionState.from_dict(redis_data)
        _SESSIONS[session_id] = session  # populate L1
        logger.debug("Session loaded from Redis: %s", session_id[:8])
        return session

    # Create new session
    session = SessionState(session_id=session_id)
    _SESSIONS[session_id] = session
    redis_client.set_session_data(session_id, session.to_dict())
    logger.debug("New session created: %s", session_id[:8])
    return session


def update(session: SessionState) -> None:
    """
    Write-through update: persist to both in-memory and Redis.
    If Redis is down, at least in-memory is updated.
    """
    _SESSIONS[session.session_id] = session
    redis_client.set_session_data(session.session_id, session.to_dict())
