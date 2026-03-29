"""
Margin Analysis Engine for Pocharlies RAG
Cross-references supplier costs, our retail prices, and competitor prices
to calculate margins and competitive position.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)


@dataclass
class MarginReport:
    product_handle: str
    product_title: str
    our_price: Optional[float]
    supplier_cost: Optional[float] = None
    supplier_slug: Optional[str] = None
    competitor_avg_price: Optional[float] = None
    competitor_prices: List[dict] = field(default_factory=list)
    supplier_prices: List[dict] = field(default_factory=list)
    margin_pct: Optional[float] = None
    competitive_position: str = "unknown"  # cheaper | at_parity | more_expensive

    def to_dict(self) -> dict:
        return {
            "product_handle": self.product_handle,
            "product_title": self.product_title,
            "our_price": self.our_price,
            "supplier_cost": self.supplier_cost,
            "best_supplier": self.supplier_slug,
            "competitor_avg_price": self.competitor_avg_price,
            "margin_pct": self.margin_pct,
            "competitive_position": self.competitive_position,
            "supplier_prices": self.supplier_prices,
            "competitor_prices": self.competitor_prices[:5],
        }


@dataclass
class MarginAlert:
    product_handle: str
    product_title: str
    alert_type: str  # low_margin | competitor_cheaper | no_supplier
    details: str
    our_price: Optional[float] = None
    supplier_cost: Optional[float] = None
    competitor_avg_price: Optional[float] = None
    margin_pct: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "product_handle": self.product_handle,
            "product_title": self.product_title,
            "alert_type": self.alert_type,
            "details": self.details,
            "our_price": self.our_price,
            "supplier_cost": self.supplier_cost,
            "competitor_avg_price": self.competitor_avg_price,
            "margin_pct": self.margin_pct,
        }


class MarginAnalyzer:
    """Calculates margins by cross-referencing supplier, own, and competitor prices."""

    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        product_indexer=None,
        supplier_indexer=None,
        competitor_indexer=None,
        embedding_model=None,
        qdrant_client=None,
    ):
        self.product_indexer = product_indexer
        self.supplier_indexer = supplier_indexer
        self.competitor_indexer = competitor_indexer
        self.model = embedding_model
        self.qdrant = qdrant_client

    def analyze_product(self, handle: str) -> Optional[MarginReport]:
        """Full margin analysis for a single product by handle."""
        # Get our product
        our_product = self._find_our_product(handle)
        if not our_product:
            return None

        title = our_product.get("title", handle)
        our_price = our_product.get("price")
        brand = our_product.get("brand", "")

        report = MarginReport(
            product_handle=handle,
            product_title=title,
            our_price=our_price,
        )

        # Find supplier prices
        query = f"{brand} {title}".strip()
        if self.supplier_indexer:
            supplier_results = self.supplier_indexer.search(query=query, top_k=5)
            for sr in supplier_results:
                wp = sr.get("wholesale_price") or sr.get("retail_price")
                if wp:
                    report.supplier_prices.append({
                        "supplier": sr.get("supplier_slug", ""),
                        "price": wp,
                        "title": sr.get("title", ""),
                        "url": sr.get("url", ""),
                    })

            if report.supplier_prices:
                cheapest = min(report.supplier_prices, key=lambda x: x["price"])
                report.supplier_cost = cheapest["price"]
                report.supplier_slug = cheapest["supplier"]

        # Find competitor prices
        if self.competitor_indexer:
            comp_results = self.competitor_indexer.search(query=query, top_k=10)
            for cr in comp_results:
                cp = cr.get("price")
                if cp and isinstance(cp, (int, float)) and cp > 0:
                    report.competitor_prices.append({
                        "domain": cr.get("domain", cr.get("source_domain", "")),
                        "price": cp,
                        "title": cr.get("title", cr.get("name", "")),
                        "url": cr.get("url", cr.get("source_url", "")),
                    })

            if report.competitor_prices:
                prices = [c["price"] for c in report.competitor_prices]
                report.competitor_avg_price = round(sum(prices) / len(prices), 2)

        # Calculate margin
        if our_price and report.supplier_cost:
            report.margin_pct = round(
                (our_price - report.supplier_cost) / our_price * 100, 1
            )

        # Competitive position
        if our_price and report.competitor_avg_price:
            delta = (our_price - report.competitor_avg_price) / report.competitor_avg_price * 100
            if delta < -5:
                report.competitive_position = "cheaper"
            elif delta > 5:
                report.competitive_position = "more_expensive"
            else:
                report.competitive_position = "at_parity"

        return report

    def analyze_brand(self, brand: str, limit: int = 50) -> dict:
        """Aggregate margin analysis for all products of a brand."""
        products = self._find_products_by_brand(brand, limit=limit)
        reports = []
        alerts = []

        for p in products:
            handle = p.get("handle", "")
            if not handle:
                continue
            report = self.analyze_product(handle)
            if report:
                reports.append(report)
                alerts.extend(self._check_alerts(report))

        # Aggregate
        margins = [r.margin_pct for r in reports if r.margin_pct is not None]
        avg_margin = round(sum(margins) / len(margins), 1) if margins else None
        low_margin_count = sum(1 for m in margins if m < 15)

        positions = [r.competitive_position for r in reports]
        pos_dist = {
            "cheaper": positions.count("cheaper"),
            "at_parity": positions.count("at_parity"),
            "more_expensive": positions.count("more_expensive"),
            "unknown": positions.count("unknown"),
        }

        return {
            "brand": brand,
            "products_analyzed": len(reports),
            "avg_margin_pct": avg_margin,
            "low_margin_count": low_margin_count,
            "competitive_position_distribution": pos_dist,
            "alerts": [a.to_dict() for a in alerts[:20]],
            "products": [r.to_dict() for r in reports],
        }

    def compare_suppliers(self, query: str, top_k: int = 10) -> List[dict]:
        """Cross-supplier price comparison for a product query."""
        if not self.supplier_indexer:
            return []

        results = self.supplier_indexer.search(query=query, top_k=top_k)
        # Group by product name similarity and sort by price
        comparison = []
        for r in results:
            wp = r.get("wholesale_price") or r.get("retail_price")
            comparison.append({
                "title": r.get("title", ""),
                "supplier": r.get("supplier_slug", ""),
                "wholesale_price": wp,
                "retail_price": r.get("retail_price"),
                "currency": r.get("currency", "EUR"),
                "url": r.get("url", ""),
                "availability": r.get("availability", ""),
            })

        comparison.sort(key=lambda x: x.get("wholesale_price") or 999999)
        return comparison

    def flag_issues(self, limit: int = 100) -> List[dict]:
        """Find products with margin issues."""
        # Get all our products
        if not self.product_indexer:
            return []

        products = self._scroll_our_products(limit=limit)
        alerts = []

        for p in products:
            handle = p.get("handle", "")
            if not handle:
                continue
            report = self.analyze_product(handle)
            if report:
                alerts.extend(self._check_alerts(report))

        return [a.to_dict() for a in alerts]

    def _check_alerts(self, report: MarginReport) -> List[MarginAlert]:
        """Check a margin report for alert conditions."""
        alerts = []

        if report.margin_pct is not None and report.margin_pct < 15:
            alerts.append(MarginAlert(
                product_handle=report.product_handle,
                product_title=report.product_title,
                alert_type="low_margin",
                details=f"Margin is only {report.margin_pct}% (supplier: {report.supplier_slug}, cost: {report.supplier_cost})",
                our_price=report.our_price,
                supplier_cost=report.supplier_cost,
                margin_pct=report.margin_pct,
            ))

        if report.competitive_position == "more_expensive" and report.competitor_avg_price:
            alerts.append(MarginAlert(
                product_handle=report.product_handle,
                product_title=report.product_title,
                alert_type="competitor_cheaper",
                details=f"Our price {report.our_price}€ vs competitor avg {report.competitor_avg_price}€",
                our_price=report.our_price,
                competitor_avg_price=report.competitor_avg_price,
            ))

        if report.our_price and not report.supplier_cost:
            alerts.append(MarginAlert(
                product_handle=report.product_handle,
                product_title=report.product_title,
                alert_type="no_supplier",
                details="No supplier price found — cannot calculate margin",
                our_price=report.our_price,
            ))

        return alerts

    def _find_our_product(self, handle: str) -> Optional[dict]:
        """Find a product in our catalog by handle."""
        if not self.qdrant:
            return None
        try:
            results = self.qdrant.scroll(
                collection_name="skirmshop_products_v2",
                scroll_filter=Filter(must=[
                    FieldCondition(key="handle", match=MatchValue(value=handle))
                ]),
                limit=1,
                with_payload=True,
            )
            points = results[0]
            return points[0].payload if points else None
        except Exception as e:
            logger.debug(f"Product lookup error for {handle}: {e}")
            return None

    def _find_products_by_brand(self, brand: str, limit: int = 50) -> List[dict]:
        """Find our products by brand."""
        if not self.qdrant:
            return []
        try:
            results = self.qdrant.scroll(
                collection_name="skirmshop_products_v2",
                scroll_filter=Filter(must=[
                    FieldCondition(key="brand", match=MatchValue(value=brand))
                ]),
                limit=limit,
                with_payload=True,
            )
            return [p.payload for p in results[0]]
        except Exception as e:
            logger.debug(f"Brand search error for {brand}: {e}")
            return []

    def _scroll_our_products(self, limit: int = 100) -> List[dict]:
        """Scroll our product catalog."""
        if not self.qdrant:
            return []
        try:
            results = self.qdrant.scroll(
                collection_name="skirmshop_products_v2",
                limit=limit,
                with_payload=True,
            )
            return [p.payload for p in results[0]]
        except Exception:
            return []
