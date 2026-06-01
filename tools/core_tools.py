"""
tools/core_tools.py
───────────────────
All 5 required deterministic tools.
RULE: These tools are the ONLY source of truth for facts.
      LLM formats/explains — never decides or invents.

Redis caching:
  • Each tool checks Redis cache BEFORE executing.
  • On cache miss, execute deterministic logic, then cache result.
  • vendor_validate is NEVER cached (unique per submission).
  • Graceful degradation: if Redis is down, tools run uncached.

Design decisions:
  • Pydantic input models for schema validation at the API boundary.
  • Every tool returns a dict with a `tool` field for traceability.
  • Latency is measured per-call (includes cache lookup time).
"""
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

import data_loader as db
import redis_client


# ── Cache-aware tool wrapper ──────────────────────────────────────

def _run_with_cache(tool_name: str, cache_args: dict, execute_fn):
    """
    Check Redis cache → return cached result on hit.
    On miss → execute tool → cache result → return.
    """
    cached = redis_client.get_cached_result(tool_name, cache_args)
    if cached is not None:
        cached["_cache_hit"] = True
        return cached

    result = execute_fn()
    redis_client.set_cached_result(tool_name, cache_args, result)
    result["_cache_hit"] = False
    return result


# ─────────────────────────────────────────────────────────────────
# TOOL 1: hot_picks
# Ranked products by popularity_score within budget, excluding
# blocked states. This is the "truth" source for recommendations.
# ─────────────────────────────────────────────────────────────────

class HotPicksInput(BaseModel):
    state: str = Field(..., description="2-letter US state code, e.g. CA")
    budget: float = Field(..., description="Max total budget in USD")
    limit: int = Field(default=5, description="Max number of products to return")


def hot_picks(state: str, budget: float, limit: int = 5) -> Dict[str, Any]:
    """
    Returns ranked products by popularity_score that fit within budget
    and are NOT blocked in the given state.
    Deterministic — no LLM involved.
    """
    state = state.upper()
    cache_args = {"state": state, "budget": budget, "limit": limit}

    def _execute():
        t0 = time.time()
        products = db.get_products()

        eligible = [
            p for p in products
            if p["price"] <= budget and state not in p.get("blocked_states", [])
        ]

        ranked = sorted(eligible, key=lambda p: p["popularity_score"], reverse=True)[:limit]

        return {
            "tool": "hot_picks",
            "state": state,
            "budget": budget,
            "results": ranked,
            "count": len(ranked),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    return _run_with_cache("hot_picks", cache_args, _execute)


# ─────────────────────────────────────────────────────────────────
# TOOL 2: compliance_filter
# Deterministic compliance gate — BLOCKED products must NEVER be
# recommended. LLM explains reasons but never decides.
# ─────────────────────────────────────────────────────────────────

class ComplianceFilterInput(BaseModel):
    state: str = Field(..., description="2-letter US state code")
    product_ids: List[int] = Field(..., description="List of product_ids to check")


def compliance_filter(state: str, product_ids: List[int]) -> Dict[str, Any]:
    """
    Deterministic compliance gate.
    Returns ALLOWED / BLOCKED / REVIEW for each product.
    BLOCKED products must NEVER be recommended. LLM explains reasons, never decides.
    """
    state = state.upper()
    cache_args = {"state": state, "product_ids": sorted(product_ids)}

    def _execute():
        t0 = time.time()
        products = {p["product_id"]: p for p in db.get_products()}

        results = []
        for pid in product_ids:
            p = products.get(pid)
            if not p:
                results.append({
                    "product_id": pid,
                    "status": "NOT_FOUND",
                    "reason_code": "PRODUCT_NOT_IN_CATALOG",
                })
                continue

            blocked = state in p.get("blocked_states", [])
            needs_lab = p.get("lab_report_required", False)

            if blocked:
                status = "BLOCKED"
                reason_code = f"STATE_RESTRICTION_{state}"
            elif needs_lab:
                status = "REVIEW"
                reason_code = "LAB_REPORT_REQUIRED"
            else:
                status = "ALLOWED"
                reason_code = "CLEAR"

            results.append({
                "product_id": pid,
                "sku": p["sku"],
                "name": p["name"],
                "status": status,
                "reason_code": reason_code,
                "blocked_states": p.get("blocked_states", []),
                "flags": p.get("flags", {}),
            })

        # Find allowed alternatives for any blocked products
        blocked_categories = set()
        for r in results:
            if r.get("status") == "BLOCKED" and r.get("name"):
                # Extract category prefix (e.g., "Nicotine Pouches" from "Nicotine Pouches - Arctic 20")
                blocked_categories.add(r["name"].split(" - ")[0])

        alternatives = []
        if blocked_categories:
            for p in db.get_products():
                cat = p["name"].split(" - ")[0]
                if (cat in blocked_categories
                        and state not in p.get("blocked_states", [])
                        and p["product_id"] not in product_ids):
                    alternatives.append({
                        "product_id": p["product_id"],
                        "sku": p["sku"],
                        "name": p["name"],
                        "price": p["price"],
                        "popularity_score": p["popularity_score"],
                    })

        return {
            "tool": "compliance_filter",
            "state": state,
            "results": results,
            "alternatives": sorted(alternatives, key=lambda x: x["popularity_score"], reverse=True)[:3],
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    return _run_with_cache("compliance_filter", cache_args, _execute)


# ─────────────────────────────────────────────────────────────────
# TOOL 3: stock_by_warehouse
# Returns warehouse-level stock quantities for a product.
# Short Redis TTL (30s) because inventory is volatile.
# ─────────────────────────────────────────────────────────────────

class StockByWarehouseInput(BaseModel):
    product_id: int = Field(..., description="Product ID to check stock for")


def stock_by_warehouse(product_id: int) -> Dict[str, Any]:
    """
    Returns warehouse-level stock quantities for a product.
    Deterministic read from seed inventory data.
    """
    cache_args = {"product_id": product_id}

    def _execute():
        t0 = time.time()
        inventory = db.get_inventory()
        product = db.get_product_by_id(product_id)

        if not product:
            return {
                "tool": "stock_by_warehouse",
                "error": "PRODUCT_NOT_FOUND",
                "product_id": product_id,
                "latency_ms": round((time.time() - t0) * 1000, 2),
            }

        warehouses = [row for row in inventory if row["product_id"] == product_id]
        total_qty = sum(w["qty"] for w in warehouses)

        return {
            "tool": "stock_by_warehouse",
            "product_id": product_id,
            "sku": product["sku"],
            "name": product["name"],
            "warehouses": warehouses,
            "total_qty": total_qty,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    return _run_with_cache("stock_by_warehouse", cache_args, _execute)


# ─────────────────────────────────────────────────────────────────
# TOOL 4: vendor_validate
# Validates vendor product submission. NEVER cached (unique per
# submission). Returns PASS / REVIEW / FAIL + missing_fields.
# ─────────────────────────────────────────────────────────────────

REQUIRED_BASE_FIELDS = ["name", "category", "net_wt_oz", "net_vol_ml"]
REQUIRED_LAB_CATEGORIES = {"THC Beverage", "THC", "CBD", "Kratom"}
REQUIRED_NICOTINE_FIELDS = ["nicotine_mg"]


class VendorValidateInput(BaseModel):
    attributes: Dict[str, Any] = Field(..., description="Product attributes JSON from vendor")


def vendor_validate(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates vendor product submission.
    Returns PASS / REVIEW / FAIL + missing_fields + required_documents.
    Deterministic logic — no LLM involvement in the decision.
    NEVER cached: each submission is unique.
    """
    t0 = time.time()
    missing_fields = []
    required_documents = []
    issues = []

    # Check base required fields
    for f in REQUIRED_BASE_FIELDS:
        val = attributes.get(f)
        if val is None or val == "" or val == 0:
            missing_fields.append(f)

    # Category-specific checks
    category = attributes.get("category", "")
    if category in REQUIRED_LAB_CATEGORIES:
        required_documents.append("lab_report")
        if not attributes.get("lab_report_attached", False):
            issues.append("LAB_REPORT_MISSING")

    # Nicotine-specific checks
    flags = attributes.get("flags", {})
    nicotine_flag = flags.get("nicotine", False) if flags else False
    has_nicotine = nicotine_flag or attributes.get("nicotine_mg", 0) > 0
    if has_nicotine:
        for f in REQUIRED_NICOTINE_FIELDS:
            if attributes.get(f) is None:
                missing_fields.append(f)
        required_documents.append("age_verification_compliance_form")

    # Determine status
    if missing_fields or "LAB_REPORT_MISSING" in issues:
        status = "FAIL" if len(missing_fields) > 2 else "REVIEW"
    else:
        status = "PASS"

    checklist = {
        "name": "✅" if "name" not in missing_fields else "❌ Missing",
        "category": "✅" if "category" not in missing_fields else "❌ Missing",
        "net_wt_oz": "✅" if "net_wt_oz" not in missing_fields else "❌ Missing",
        "net_vol_ml": "✅" if "net_vol_ml" not in missing_fields else "❌ Missing",
        "lab_report": (
            "✅" if "lab_report" not in required_documents or attributes.get("lab_report_attached")
            else "❌ Required but not attached"
        ),
    }

    return {
        "tool": "vendor_validate",
        "status": status,
        "missing_fields": missing_fields,
        "issues": issues,
        "required_documents": required_documents,
        "checklist": checklist,
        "submitted_attributes": {k: v for k, v in attributes.items() if k != "raw"},
        "latency_ms": round((time.time() - t0) * 1000, 2),
        "_cache_hit": False,  # Never cached
    }


# ─────────────────────────────────────────────────────────────────
# TOOL 5: kb_search
# Naive keyword search over kb_docs filtered by visibility.
# Visibility rules enforce data boundaries:
#   internal_sales → internal + public
#   portal_vendor  → vendor + public
#   portal_customer → public only
# ─────────────────────────────────────────────────────────────────

class KbSearchInput(BaseModel):
    query: str = Field(..., description="User query to search knowledge base")
    user_type: str = Field(default="internal_sales", description="Controls visibility filtering")


def kb_search(query: str, user_type: str = "internal_sales") -> Dict[str, Any]:
    """
    Naive keyword search over kb_docs.
    Filters by doc visibility based on user_type.
    Returns top 3 short snippets — never full dumps.
    """
    cache_args = {"query": query.lower().strip(), "user_type": user_type}

    def _execute():
        t0 = time.time()
        docs = db.get_kb_docs()
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) > 3]

        # Visibility rules
        visibility_map = {
            "internal_sales": {"internal", "public"},
            "portal_vendor": {"vendor", "public"},
            "portal_customer": {"public"},
        }
        allowed_visibility = visibility_map.get(user_type, {"public"})

        scored = []
        for doc in docs:
            if doc.get("visibility") not in allowed_visibility:
                continue
            text_lower = doc["text"].lower()
            title_lower = doc["title"].lower()
            score = sum(1 for kw in keywords if kw in text_lower or kw in title_lower)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [
            {"doc_id": d["doc_id"], "title": d["title"], "snippet": d["text"][:200]}
            for _, d in scored[:3]
        ]

        return {
            "tool": "kb_search",
            "query": query,
            "results": top,
            "count": len(top),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    return _run_with_cache("kb_search", cache_args, _execute)
