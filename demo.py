"""
demo.py
───────
Runs all 5 live demo queries from the test spec IN-PROCESS.
No server required — invokes the LangGraph directly.

Usage:
  python demo.py

Requires:
  • GROQ_API_KEY in .env or environment
  • Redis running on localhost:6379 (optional — degrades gracefully)
"""
import json
import sys
import os
import uuid

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from data_loader import load_seed_data
from chains.orchestrator import get_graph
from state.session import get_or_create
from observability import RequestTrace, setup_logging
import redis_client


# ── ANSI colors for terminal output ──────────────────────────────

class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def banner(text: str) -> str:
    return f"{C.BOLD}{C.HEADER}{text}{C.RESET}"


def label(text: str) -> str:
    return f"{C.BOLD}{C.CYAN}{text}{C.RESET}"


def success(text: str) -> str:
    return f"{C.GREEN}{text}{C.RESET}"


def warn(text: str) -> str:
    return f"{C.YELLOW}{text}{C.RESET}"


# ── Query runner ──────────────────────────────────────────────────

def run_query(query: str, session_id: str, user_type: str = "internal_sales") -> dict:
    session = get_or_create(session_id)
    trace = RequestTrace()
    trace.user_type = user_type
    trace.raw_query = query

    graph = get_graph()
    result = graph.invoke({
        "query": query,
        "user_type": user_type,
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

    return {
        "intent": result["intent"],
        "response": result["final_response"],
        "error": result.get("error"),
        "trace": trace,
    }


# ── Main demo ─────────────────────────────────────────────────────

def main():
    # Initialize
    setup_logging()
    load_seed_data("data/seed_data.json")
    redis_ok = redis_client.init_redis()

    session_id = str(uuid.uuid4())

    print()
    print(banner("=" * 72))
    print(banner("  REELO AI — LIVE DEMO (5 Canonical Queries)"))
    print(banner(f"  Session: {session_id[:8]}..."))
    print(banner(f"  Redis:   {'[OK] connected' if redis_ok else '[WARN] unavailable (running without cache)'}"))
    print(banner("=" * 72))

    queries = [
        {
            "num": 1,
            "label": "SALES — Hot picks for CA under $5000",
            "query": "Give me hot picks for CA under $5000",
            "user_type": "internal_sales",
        },
        {
            "num": 2,
            "label": "COMPLIANCE — Why is SKU-1006 blocked in CA?",
            "query": "Why is SKU-1006 not available in CA? Suggest alternatives.",
            "user_type": "internal_sales",
        },
        {
            "num": 3,
            "label": "OPS — Stock levels for SKU-1005",
            "query": "How much stock does SKU-1005 have and where?",
            "user_type": "internal_sales",
        },
        {
            "num": 4,
            "label": "VENDOR — Missing fields check",
            "query": "I'm uploading a product missing Net Wt and no lab report — what do I fix?",
            "user_type": "portal_vendor",
        },
        {
            "num": 5,
            "label": "BASKET — Add 2 of the first one",
            "query": "Ok add 2 of the first one to the basket",
            "user_type": "internal_sales",
        },
    ]

    for q in queries:
        print()
        print(f"{C.BOLD}{C.BLUE}{'─' * 72}{C.RESET}")
        lbl = label(f"[Q{q['num']}]")
        print(f"  {lbl} {q['label']}")
        print(f"  {C.DIM}user_type={q['user_type']}{C.RESET}")
        print(f"  {C.DIM}query: \"{q['query']}\"{C.RESET}")
        print(f"{C.BOLD}{C.BLUE}{'─' * 72}{C.RESET}")

        r = run_query(q["query"], session_id, q["user_type"])

        print(f"  {label('Intent:')} {success(r['intent'])}")

        # Show tool calls from trace
        if r["trace"].tools_called:
            print(f"  {label('Tools:')}")
            for tc in r["trace"].tools_called:
                cache = f" {warn('(cached)')}" if tc.cache_hit else ""
                print(f"    → {tc.tool_name}({json.dumps(tc.args, default=str)[:60]}) "
                      f"[{tc.latency_ms}ms]{cache}")

        print(f"  {label('Response:')}")
        # Indent response lines
        for line in r["response"][:700].split("\n"):
            print(f"    {line}")

        if r["error"]:
            print(f"  {C.RED}Error: {r['error']}{C.RESET}")

    print()
    print(banner("=" * 72))
    print(banner("  Demo complete."))
    print(banner("  Check [TRACE] JSON logs above for full observability data."))
    print(banner("=" * 72))
    print()


if __name__ == "__main__":
    main()
