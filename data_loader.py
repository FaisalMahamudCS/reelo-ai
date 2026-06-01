"""
data_loader.py
─────────────
Loads seed_data.json into memory at startup.
All tools read from these in-memory structures — never from LLM context.

Design decision: Pydantic models validate data at load time so we fail
fast on corrupt seed files rather than discovering issues mid-request.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


# ── Pydantic schemas for seed data validation ──────────────────────

class ProductFlags(BaseModel):
    nicotine: bool = False
    thc: bool = False
    cbd: bool = False
    kratom: bool = False
    mushroom: bool = False


class Product(BaseModel):
    product_id: int
    sku: str
    name: str
    category: str
    flags: ProductFlags
    blocked_states: List[str] = Field(default_factory=list)
    lab_report_required: bool = False
    price: float
    popularity_score: float


class InventoryRow(BaseModel):
    product_id: int
    warehouse: str
    qty: int


class Customer(BaseModel):
    customer_id: str
    name: str
    state: str
    tier: str


class Vendor(BaseModel):
    vendor_id: str
    name: str
    status: str


class KBDoc(BaseModel):
    doc_id: str
    title: str
    visibility: str
    text: str


class SeedData(BaseModel):
    """Top-level schema — validates entire seed file at load time."""
    generated_at: Optional[str] = None
    products: List[Product]
    inventory: List[InventoryRow]
    customers: List[Customer]
    vendors: List[Vendor]
    kb_docs: List[KBDoc]


# ── In-memory store ────────────────────────────────────────────────

_DATA: Optional[SeedData] = None


def load_seed_data(path: str = "data/seed_data.json") -> None:
    """
    Load and validate seed data at startup.
    Raises pydantic.ValidationError on malformed data — fail fast.
    """
    global _DATA
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _DATA = SeedData(**raw)


def _ensure_loaded() -> SeedData:
    if _DATA is None:
        raise RuntimeError("Seed data not loaded. Call load_seed_data() at startup.")
    return _DATA


# ── Typed accessors (tools call these, never access _DATA directly) ─

def get_products() -> List[Dict[str, Any]]:
    return [p.model_dump() for p in _ensure_loaded().products]


def get_inventory() -> List[Dict[str, Any]]:
    return [i.model_dump() for i in _ensure_loaded().inventory]


def get_customers() -> List[Dict[str, Any]]:
    return [c.model_dump() for c in _ensure_loaded().customers]


def get_vendors() -> List[Dict[str, Any]]:
    return [v.model_dump() for v in _ensure_loaded().vendors]


def get_kb_docs() -> List[Dict[str, Any]]:
    return [d.model_dump() for d in _ensure_loaded().kb_docs]


def get_product_by_sku(sku: str) -> Optional[Dict[str, Any]]:
    for p in _ensure_loaded().products:
        if p.sku.upper() == sku.upper():
            return p.model_dump()
    return None


def get_product_by_id(product_id: int) -> Optional[Dict[str, Any]]:
    for p in _ensure_loaded().products:
        if p.product_id == product_id:
            return p.model_dump()
    return None
