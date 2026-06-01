"""
redis_client.py
───────────────
Redis connection manager with connection pooling and graceful degradation.

Design tradeoffs:
  • Connection pool: reuse connections across requests (avoid per-request overhead).
  • Graceful degradation: if Redis is unreachable, tools still execute (just uncached).
    We log a warning and fall back — never fail a request because cache is down.
  • TTL strategy: per-tool configuration so volatile data (stock) expires fast while
    stable data (compliance rules) lives longer.
  • Key namespace: `reelo:{tool}:{arg_hash}` — deterministic, collision-free.

Production notes:
  • In K8s: use Redis Sentinel or Cluster for HA.
  • Consider read replicas for geographically distributed deployments.
"""
import os
import json
import hashlib
import logging
from typing import Any, Optional

logger = logging.getLogger("reelo.redis")

_redis_client = None
_redis_available = False


def init_redis(url: Optional[str] = None) -> bool:
    """
    Initialize Redis connection pool.
    Returns True if connected, False if Redis is unavailable (graceful degradation).
    """
    global _redis_client, _redis_available
    redis_url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis
        _redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=2,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        _redis_client.ping()
        _redis_available = True
        logger.info("Redis connected: %s", redis_url)
        return True
    except Exception as e:
        _redis_available = False
        logger.warning("Redis unavailable (%s). Running without cache — tools will execute uncached.", e)
        return False


def close_redis() -> None:
    global _redis_client, _redis_available
    if _redis_client:
        try:
            _redis_client.close()
        except Exception:
            pass
    _redis_client = None
    _redis_available = False


def _make_key(tool_name: str, args: dict) -> str:
    """Deterministic cache key: reelo:{tool}:{sha256(sorted_args)[:12]}"""
    arg_str = json.dumps(args, sort_keys=True, default=str)
    arg_hash = hashlib.sha256(arg_str.encode()).hexdigest()[:12]
    return f"reelo:{tool_name}:{arg_hash}"


# ── Tool result caching ────────────────────────────────────────────

# TTL per tool (seconds). Tradeoff:
#   stock → 30s  (inventory changes fast, accept brief staleness)
#   hot_picks → 300s  (product catalog is relatively stable)
#   compliance → 600s  (regulations change rarely)
#   kb_search → 300s  (knowledge base updates infrequently)
#   vendor_validate → 0  (unique per submission, never cache)
TOOL_TTL = {
    "hot_picks": 300,
    "compliance_filter": 600,
    "stock_by_warehouse": 30,
    "kb_search": 300,
    "vendor_validate": 0,
}


def get_cached_result(tool_name: str, args: dict) -> Optional[dict]:
    """
    Check Redis for cached tool result.
    Returns None on miss or if Redis is unavailable (graceful degradation).
    """
    if not _redis_available or TOOL_TTL.get(tool_name, 0) == 0:
        return None
    try:
        key = _make_key(tool_name, args)
        raw = _redis_client.get(key)
        if raw:
            logger.debug("Cache HIT: %s", key)
            return json.loads(raw)
        logger.debug("Cache MISS: %s", key)
        return None
    except Exception as e:
        logger.warning("Redis GET failed (%s). Proceeding uncached.", e)
        return None


def set_cached_result(tool_name: str, args: dict, result: dict) -> None:
    """
    Cache tool result in Redis with appropriate TTL.
    No-op if Redis unavailable or TTL is 0 (never cache).
    """
    ttl = TOOL_TTL.get(tool_name, 0)
    if not _redis_available or ttl == 0:
        return
    try:
        key = _make_key(tool_name, args)
        _redis_client.setex(key, ttl, json.dumps(result, default=str))
        logger.debug("Cache SET: %s (TTL=%ds)", key, ttl)
    except Exception as e:
        logger.warning("Redis SET failed (%s). Result not cached.", e)


# ── Session storage in Redis ───────────────────────────────────────

SESSION_TTL = 3600  # 1 hour


def get_session_data(session_id: str) -> Optional[dict]:
    """Retrieve session from Redis. Returns None if not found or Redis unavailable."""
    if not _redis_available:
        return None
    try:
        key = f"reelo:session:{session_id}"
        raw = _redis_client.get(key)
        if raw:
            return json.loads(raw)
        return None
    except Exception as e:
        logger.warning("Redis session GET failed: %s", e)
        return None


def set_session_data(session_id: str, data: dict) -> None:
    """Store session in Redis with TTL. No-op if Redis unavailable."""
    if not _redis_available:
        return
    try:
        key = f"reelo:session:{session_id}"
        _redis_client.setex(key, SESSION_TTL, json.dumps(data, default=str))
    except Exception as e:
        logger.warning("Redis session SET failed: %s", e)


def is_available() -> bool:
    """Check if Redis is currently available."""
    return _redis_available
