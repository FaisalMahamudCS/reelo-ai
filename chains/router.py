"""
chains/router.py
────────────────
Intent classification: keyword-first (zero cost), LLM fallback only if ambiguous.

Intents: SALES_RECO | COMPLIANCE_CHECK | VENDOR_ONBOARDING |
         OPS_STOCK | GENERAL_KB | BASKET_FOLLOWUP

Design decisions:
  • Regex matching is O(n_patterns) per query — microsecond latency, zero tokens.
  • Score-based: multiple patterns can match; highest score wins. This handles
    ambiguous queries like "why can't I buy this in CA" (compliance > sales).
  • BASKET_FOLLOWUP detection happens BEFORE regex: it checks for basket
    keywords AND requires prior product context in session. Without context,
    it falls through to regex (prevents false positives).
  • LLM fallback via Groq: only fires when NO regex matches. Costs ~50 tokens
    per classification — still 100x cheaper than sending the full query.
  • Extract helpers (state, budget, SKU, quantity) are pure regex — no LLM.
"""
import os
import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger("reelo.router")


# ── Intent patterns ───────────────────────────────────────────────
# Each intent has a list of regex patterns. Query is scored against all
# patterns; intent with highest cumulative score wins.

INTENT_PATTERNS = {
    "SALES_RECO": [
        r"\bhot picks?\b", r"\bbest sellers?\b", r"\btop products?\b",
        r"\bunder \$[\d,]+\b", r"\brecommend\b", r"\bpopular\b",
        r"\bwhat (should|can) (i|we) (sell|order)\b",
        r"\bbudget\b", r"\baffordable\b",
    ],
    "COMPLIANCE_CHECK": [
        r"\bblocked?\b", r"\billegal\b", r"\bnot available\b",
        r"\bcomplian\w*\b", r"\brestrict\w*\b", r"\bwhy (is|isn'?t|can'?t)\b",
        r"\ballowed?\b", r"\bpermit\w*\b", r"\bstate law\b",
        r"\bnot (available|sold|allowed)\b",
    ],
    "VENDOR_ONBOARDING": [
        r"\bvendor\b", r"\bonboard\w*\b", r"\bupload\w*\b",
        r"\bsubmit\w*\b", r"\bcatalog\b", r"\blab report\b",
        r"\bnet wt\b", r"\bnet vol\b", r"\bmissing field\b",
        r"\bwhat do i fix\b", r"\bproduct form\b",
        r"\bmissing\b.*\b(wt|weight|vol|report)\b",
    ],
    "OPS_STOCK": [
        r"\bstock\b", r"\binventory\b", r"\bwarehouse\b",
        r"\bqty\b", r"\bquantit\w*\b", r"\bhow much\b.*\bhave\b",
        r"\bavailab\w*\b.*\bwhere\b", r"\bhow many\b",
    ],
    "GENERAL_KB": [
        r"\bpolic\w*\b", r"\bsop\b", r"\bprocedure\b",
        r"\breturn\b", r"\bshipping\b", r"\bguide\b",
        r"\bhow do (i|we)\b", r"\bwhat is\b", r"\bexplain\b",
    ],
    "BASKET_FOLLOWUP": [
        r"\badd\b.*\b(basket|cart)\b", r"\bbasket\b", r"\bcart\b",
        r"\bfirst one\b", r"\bthat one\b", r"\bsecond one\b",
        r"\bput\b.*\b(in|to)\b.*\b(basket|cart|order)\b",
    ],
}

# Compile once at import time
_COMPILED = {
    intent: [re.compile(p, re.IGNORECASE) for p in patterns]
    for intent, patterns in INTENT_PATTERNS.items()
}


def classify_intent(query: str) -> Tuple[str, str]:
    """
    Returns (intent, method) where method is 'keyword' or 'llm_fallback'.
    Keyword matching costs zero tokens. LLM fallback only if truly ambiguous.
    """
    scores = {intent: 0 for intent in _COMPILED}

    for intent, patterns in _COMPILED.items():
        for pattern in patterns:
            if pattern.search(query):
                scores[intent] += 1

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    if best_score > 0:
        logger.debug("Intent classified via keyword: %s (score=%d)", best_intent, best_score)
        return best_intent, "keyword"

    # LLM fallback: classify via Groq (costs ~50 tokens)
    return _llm_classify_intent(query)


def _llm_classify_intent(query: str) -> Tuple[str, str]:
    """
    Fallback: use Groq LLM to classify intent when regex fails.
    Bounded to minimal tokens. Returns (intent, 'llm_fallback').
    """
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=20,  # We only need one word back
            temperature=0,
        )

        messages = [
            SystemMessage(content=(
                "Classify the user query into exactly ONE intent. "
                "Reply with ONLY the intent name, nothing else.\n"
                "Valid intents: SALES_RECO, COMPLIANCE_CHECK, VENDOR_ONBOARDING, "
                "OPS_STOCK, GENERAL_KB, BASKET_FOLLOWUP"
            )),
            HumanMessage(content=query),
        ]

        response = llm.invoke(messages)
        intent = response.content.strip().upper()

        valid_intents = {
            "SALES_RECO", "COMPLIANCE_CHECK", "VENDOR_ONBOARDING",
            "OPS_STOCK", "GENERAL_KB", "BASKET_FOLLOWUP",
        }
        if intent in valid_intents:
            logger.info("Intent classified via LLM fallback: %s", intent)
            return intent, "llm_fallback"

        logger.warning("LLM returned invalid intent '%s', defaulting to GENERAL_KB", intent)
        return "GENERAL_KB", "llm_fallback"

    except Exception as e:
        logger.error("LLM fallback failed: %s. Defaulting to GENERAL_KB", e)
        return "GENERAL_KB", "default_fallback"


# ── Query extraction helpers (pure regex, zero LLM cost) ──────────

def extract_state_from_query(query: str) -> Optional[str]:
    """Extract US state code or name from query."""
    state_map = {
        "california": "CA", "texas": "TX", "florida": "FL",
        "new york": "NY", "massachusetts": "MA", "idaho": "ID",
        "utah": "UT", "alabama": "AL", "arkansas": "AR",
        "indiana": "IN", "rhode island": "RI", "vermont": "VT",
        "wisconsin": "WI", "kansas": "KS", "south dakota": "SD",
        "wyoming": "WY", "nebraska": "NE",
    }
    query_lower = query.lower()

    # Try 2-letter code first (e.g., "CA", "TX")
    match = re.search(r"\b([A-Z]{2})\b", query)
    if match:
        return match.group(1)

    # Try full state name
    for name, code in state_map.items():
        if name in query_lower:
            return code

    return None


def extract_budget_from_query(query: str) -> Optional[float]:
    """Extract dollar amount from query."""
    match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", query)
    if match:
        return float(match.group(1).replace(",", ""))
    match = re.search(r"([\d,]+)\s*(?:dollars?|usd)", query, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def extract_sku_from_query(query: str) -> Optional[str]:
    """Extract SKU from query."""
    match = re.search(r"SKU-\d+", query, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return None


def extract_quantity_from_query(query: str) -> Optional[int]:
    """Extract quantity for basket operations."""
    match = re.search(r"\badd\s+(\d+)\b", query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d+)\s+of\b", query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_ordinal_from_query(query: str) -> Optional[int]:
    """Extract ordinal reference like 'first one', 'second one', 'third one'."""
    ordinals = {
        "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
        "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4,
    }
    query_lower = query.lower()
    for word, idx in ordinals.items():
        if word in query_lower:
            return idx
    return 0  # Default to first item
