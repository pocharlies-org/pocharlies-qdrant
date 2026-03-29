"""
Purchase Recommendation Engine for Pocharlies RAG
Gap analysis, restock recommendations, and Picqer WMS integration.
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)


@dataclass
class GapProduct:
    """Product competitors have that we don't."""
    title: str
    brand: str
    category: str
    competitor_count: int
    competitor_avg_price: float
    competitor_domains: List[str]
    available_from_suppliers: List[dict]
    opportunity_score: float  # competitor_count * avg_price

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "brand": self.brand,
            "category": self.category,
            "competitor_count": self.competitor_count,
            "competitor_avg_price": self.competitor_avg_price,
            "competitor_domains": self.competitor_domains[:5],
            "available_from_suppliers": self.available_from_suppliers[:5],
            "opportunity_score": self.opportunity_score,
        }


@dataclass
class RestockRecommendation:
    product_handle: str
    product_title: str
    brand: str
    our_price: float
    supplier_cost: Optional[float]
    best_supplier: Optional[str]
    margin_pct: Optional[float]
    picqer_stock: Optional[int]
    competitor_has_stock: bool
    priority: str  # high | medium | low

    def to_dict(self) -> dict:
        return {
            "product_handle": self.product_handle,
            "product_title": self.product_title,
            "brand": self.brand,
            "our_price": self.our_price,
            "supplier_cost": self.supplier_cost,
            "best_supplier": self.best_supplier,
            "margin_pct": self.margin_pct,
            "picqer_stock": self.picqer_stock,
            "competitor_has_stock": self.competitor_has_stock,
            "priority": self.priority,
        }


class PurchaseRecommender:
    """Generates purchase recommendations by combining margin analysis, gap analysis, and Picqer stock data."""

    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        margin_analyzer=None,
        supplier_registry=None,
        supplier_indexer=None,
        product_indexer=None,
        qdrant_client=None,
        embedding_model=None,
    ):
        self.margin_analyzer = margin_analyzer
        self.supplier_registry = supplier_registry
        self.supplier_indexer = supplier_indexer
        self.product_indexer = product_indexer
        self.qdrant = qdrant_client
        self.model = embedding_model

    def gap_analysis(self, limit: int = 50) -> List[dict]:
        """Find products competitors sell that we don't have."""
        if not self.qdrant or not self.model:
            return []

        # Get unique competitor products
        competitor_products = self._scroll_competitor_products(limit=500)
        our_handles = self._get_our_handles()

        gaps = {}
        for cp in competitor_products:
            title = cp.get("title", cp.get("name", ""))
            brand = cp.get("brand", "")
            if not title:
                continue

            # Check if we already have this product
            query = f"{brand} {title}".strip()
            match = self._find_in_our_catalog(query)
            if match:
                continue  # We have it

            # Group by normalized product name
            key = f"{brand}:{title}".lower().strip()
            if key not in gaps:
                gaps[key] = {
                    "title": title,
                    "brand": brand,
                    "category": cp.get("category", ""),
                    "prices": [],
                    "domains": set(),
                }

            price = cp.get("price")
            if price and isinstance(price, (int, float)):
                gaps[key]["prices"].append(price)
            domain = cp.get("domain", cp.get("source_domain", ""))
            if domain:
                gaps[key]["domains"].add(domain)

        # Score and rank
        gap_products = []
        for key, data in gaps.items():
            if len(data["domains"]) < 1:
                continue

            avg_price = sum(data["prices"]) / len(data["prices"]) if data["prices"] else 0
            score = len(data["domains"]) * avg_price

            # Check if available from any supplier
            supplier_options = []
            if self.supplier_indexer:
                sr = self.supplier_indexer.search(
                    query=f"{data['brand']} {data['title']}",
                    top_k=3,
                )
                for s in sr:
                    wp = s.get("wholesale_price") or s.get("retail_price")
                    if wp:
                        supplier_options.append({
                            "supplier": s.get("supplier_slug", ""),
                            "price": wp,
                            "title": s.get("title", ""),
                        })

            gap_products.append(GapProduct(
                title=data["title"],
                brand=data["brand"],
                category=data["category"],
                competitor_count=len(data["domains"]),
                competitor_avg_price=round(avg_price, 2),
                competitor_domains=list(data["domains"]),
                available_from_suppliers=supplier_options,
                opportunity_score=round(score, 2),
            ))

        gap_products.sort(key=lambda x: x.opportunity_score, reverse=True)
        return [g.to_dict() for g in gap_products[:limit]]

    def restock_recommendations(self, limit: int = 50) -> List[dict]:
        """Recommend products to restock based on margins and competitor availability."""
        if not self.margin_analyzer:
            return []

        products = self._scroll_our_products(limit=200)
        recommendations = []

        for p in products:
            handle = p.get("handle", "")
            if not handle:
                continue

            report = self.margin_analyzer.analyze_product(handle)
            if not report or not report.our_price:
                continue

            # Check Picqer stock
            picqer_stock = self._get_picqer_stock(handle)

            # Check if competitors have stock
            comp_has_stock = len(report.competitor_prices) > 0

            # Determine priority
            margin = report.margin_pct
            priority = "low"
            if margin is not None and margin >= 25 and picqer_stock is not None and picqer_stock <= 2 and comp_has_stock:
                priority = "high"
            elif margin is not None and margin >= 15 and picqer_stock is not None and picqer_stock <= 5:
                priority = "medium"

            if priority in ("high", "medium"):
                recommendations.append(RestockRecommendation(
                    product_handle=handle,
                    product_title=report.product_title,
                    brand=p.get("brand", ""),
                    our_price=report.our_price,
                    supplier_cost=report.supplier_cost,
                    best_supplier=report.supplier_slug,
                    margin_pct=margin,
                    picqer_stock=picqer_stock,
                    competitor_has_stock=comp_has_stock,
                    priority=priority,
                ))

        recommendations.sort(key=lambda x: (0 if x.priority == "high" else 1, -(x.margin_pct or 0)))
        return [r.to_dict() for r in recommendations[:limit]]

    def delist_candidates(self, limit: int = 50) -> List[dict]:
        """Products that might not be worth keeping — low margin + competitors cheaper."""
        if not self.margin_analyzer:
            return []

        products = self._scroll_our_products(limit=200)
        candidates = []

        for p in products:
            handle = p.get("handle", "")
            if not handle:
                continue

            report = self.margin_analyzer.analyze_product(handle)
            if not report:
                continue

            margin = report.margin_pct
            if margin is not None and margin < 10 and report.competitive_position == "more_expensive":
                candidates.append({
                    "product_handle": handle,
                    "product_title": report.product_title,
                    "our_price": report.our_price,
                    "supplier_cost": report.supplier_cost,
                    "competitor_avg_price": report.competitor_avg_price,
                    "margin_pct": margin,
                    "reason": f"Margin {margin}% and competitors avg {report.competitor_avg_price}€ vs our {report.our_price}€",
                })

        return candidates[:limit]

    def generate_purchase_order(self, supplier_slug: str, product_handles: List[str]) -> dict:
        """Generate a purchase order summary for a supplier."""
        if not self.supplier_registry or not self.margin_analyzer:
            return {"error": "Not initialized"}

        supplier = self.supplier_registry.get_supplier(supplier_slug)
        if not supplier:
            return {"error": f"Supplier not found: {supplier_slug}"}

        items = []
        total = 0.0

        for handle in product_handles:
            report = self.margin_analyzer.analyze_product(handle)
            if not report:
                continue

            # Find this supplier's price for the product
            supplier_price = None
            for sp in report.supplier_prices:
                if sp.get("supplier") == supplier_slug:
                    supplier_price = sp["price"]
                    break

            if supplier_price:
                items.append({
                    "handle": handle,
                    "title": report.product_title,
                    "wholesale_price": supplier_price,
                    "our_retail_price": report.our_price,
                    "margin_pct": round((report.our_price - supplier_price) / report.our_price * 100, 1) if report.our_price else None,
                })
                total += supplier_price

        return {
            "supplier": supplier.get("name", supplier_slug),
            "supplier_slug": supplier_slug,
            "currency": supplier.get("currency", "EUR"),
            "minimum_order": supplier.get("minimum_order", 0),
            "items_count": len(items),
            "total_wholesale": round(total, 2),
            "meets_minimum": total >= supplier.get("minimum_order", 0),
            "items": items,
        }

    def _get_picqer_stock(self, handle: str) -> Optional[int]:
        """Get stock level from Picqer WMS via mcporter."""
        try:
            result = subprocess.run(
                ["mcporter", "call", "picqer-wms.get_products",
                 f"search={handle}", "--output", "json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                products = data if isinstance(data, list) else data.get("data", [])
                for p in products:
                    stock = p.get("stock", {})
                    if isinstance(stock, dict):
                        return stock.get("freestock", 0)
                    return int(stock) if stock else 0
        except Exception as e:
            logger.debug(f"Picqer lookup failed for {handle}: {e}")
        return None

    def _find_in_our_catalog(self, query: str) -> Optional[dict]:
        """Check if we have a matching product."""
        if not self.qdrant or not self.model:
            return None
        try:
            prefixed = f"{self.BGE_QUERY_PREFIX}{query}"
            dense_emb = self.model.encode(prefixed, normalize_embeddings=True).tolist()
            results = self.qdrant.search(
                collection_name="skirmshop_products_v2",
                query_vector=("dense", dense_emb),
                limit=1,
                with_payload=True,
            )
            if results and results[0].score >= 0.85:
                return results[0].payload
        except Exception:
            pass
        return None

    def _get_our_handles(self) -> set:
        """Get set of all our product handles."""
        handles = set()
        try:
            offset = None
            while True:
                results = self.qdrant.scroll(
                    collection_name="skirmshop_products_v2",
                    limit=100,
                    offset=offset,
                    with_payload=["handle"],
                )
                for p in results[0]:
                    h = p.payload.get("handle", "")
                    if h:
                        handles.add(h)
                if not results[1]:
                    break
                offset = results[1]
        except Exception:
            pass
        return handles

    def _scroll_competitor_products(self, limit: int = 500) -> List[dict]:
        """Scroll competitor products."""
        try:
            results = self.qdrant.scroll(
                collection_name="competitor_products_v2",
                limit=limit,
                with_payload=True,
            )
            return [p.payload for p in results[0]]
        except Exception:
            return []

    def _scroll_our_products(self, limit: int = 200) -> List[dict]:
        """Scroll our products."""
        try:
            results = self.qdrant.scroll(
                collection_name="skirmshop_products_v2",
                limit=limit,
                with_payload=True,
            )
            return [p.payload for p in results[0]]
        except Exception:
            return []
