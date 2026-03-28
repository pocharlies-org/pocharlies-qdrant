"""Fast product extraction from crawled content — structured data, no LLM needed."""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Common airsoft brand patterns for brand detection
BRAND_PATTERNS = [
    "Tokyo Marui", "VFC", "G&G", "Specna Arms", "KWA", "KWC", "WE",
    "ASG", "Umarex", "Cyma", "Cybergun", "Classic Army", "ICS",
    "Krytac", "Ares", "S&T", "Modify", "Maxx", "Prometheus", "PDI",
    "Maple Leaf", "Action Army", "Novritsch", "Silverback",
    "Wolverine", "PolarStar", "Mancraft", "Retro Arms", "Gate",
    "Perun", "Warhead", "ZCI", "SHS", "Lonex", "Guarder",
    "Laylax", "Nine Ball", "Angel Custom", "Matrix", "JG",
    "Double Bell", "E&L", "LCT", "Real Sword", "GHK", "KJ Works",
    "Poseidon", "Maple Leaf", "T-N.T.", "Begadi", "Nuprol",
    "Evolution", "King Arms", "Bolt", "WELL", "Snow Wolf",
    "A&K", "Golden Eagle", "Double Eagle", "STTI", "KSC",
    "Maruyama", "Tanaka", "Marushin", "EMG", "AW Custom",
]


def _detect_category(title: str, text: str = "") -> Optional[str]:
    """Detect product category from title and text."""
    combined = (title + " " + text).lower()
    categories = {
        "aeg": ["aeg", "electric gun", "fusil electrico", "replica electrica"],
        "gbb": ["gbb", "gas blowback", "pistola gas"],
        "sniper": ["sniper", "bolt action", "francotirador"],
        "shotgun": ["shotgun", "escopeta"],
        "smg": ["smg", "subfusil", "submachine"],
        "magazine": ["magazine", "cargador", "mag "],
        "battery": ["battery", "bateria", "lipo", "nimh"],
        "optic": ["scope", "red dot", "sight", "mira", "optic", "visor"],
        "barrel": ["barrel", "cañon", "inner barrel", "outer barrel"],
        "hop-up": ["hop-up", "hopup", "hop up", "bucking"],
        "gear": ["gear", "engranaje", "piston", "cylinder"],
        "motor": ["motor"],
        "handguard": ["handguard", "guardamanos", "rail", "ris", "m-lok", "keymod"],
        "stock": ["stock", "culata"],
        "grip": ["grip", "empuñadura"],
        "suppressor": ["suppressor", "silencer", "silenciador"],
        "bbs": ["bbs", "balines", "0.20g", "0.25g", "0.28g", "0.30g"],
        "grenade": ["grenade", "granada", "launcher"],
        "clothing": ["uniform", "plate carrier", "vest", "chaleco", "camo"],
        "protection": ["goggles", "mask", "mascara", "gafas", "proteccion"],
    }
    for cat, keywords in categories.items():
        if any(kw in combined for kw in keywords):
            return cat
    return None


class FastProductExtractor:
    """Extract product data from Qdrant collections without LLM calls."""

    def __init__(self, qdrant_client=None):
        self.client = qdrant_client

    def extract_products(self, domain: str, collection_name: str) -> List[Dict]:
        """Extract all products from a given domain in a collection."""
        if not self.client:
            return []

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
                ),
                limit=1000,
                with_payload=True,
            )[0]

            products = []
            for point in results:
                payload = point.payload or {}
                title = payload.get("title", "")
                text = payload.get("text", "")

                # Detect brand
                brand = ""
                text_lower = (text + " " + title).lower()
                for b in BRAND_PATTERNS:
                    if b.lower() in text_lower:
                        brand = b
                        break

                # Detect category
                category = _detect_category(title, text)

                # Extract price
                price = payload.get("price")
                if not price:
                    price_match = re.search(
                        r'(\d{1,5}[.,]\d{2})\s*€|€\s*(\d{1,5}[.,]\d{2})', text
                    )
                    if price_match:
                        ps = price_match.group(1) or price_match.group(2)
                        try:
                            price = float(ps.replace(",", "."))
                        except ValueError:
                            pass

                products.append({
                    "title": title,
                    "url": payload.get("url", ""),
                    "brand": brand,
                    "category": category,
                    "price": price,
                    "domain": domain,
                    "text": text[:500],
                })

            logger.info(f"Extracted {len(products)} products from {domain} in {collection_name}")
            return products

        except Exception as e:
            logger.error(f"Fast extraction failed for {domain}: {e}")
            return []
