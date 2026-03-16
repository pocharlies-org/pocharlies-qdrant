"""Analyzes products to determine platform compatibility using keyword detection + LLM fallback."""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable

from openai import OpenAI

from compatibility_data import (
    PLATFORMS, UPGRADE_TYPES, CROSS_PLATFORM_CATEGORIES, BASE_PLATFORM_INDICATORS,
)

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityResult:
    compatible_platforms: List[str] = field(default_factory=list)
    upgrade_type: Optional[str] = None
    upgrade_priority: Optional[str] = None  # essential | recommended | optional
    is_base_platform: bool = False
    confidence: str = "low"  # high | medium | low
    reasoning: str = ""

    def to_payload(self) -> Dict[str, Any]:
        """Return fields suitable for Qdrant payload update."""
        return {
            "compatible_platforms": self.compatible_platforms,
            "upgrade_type": self.upgrade_type,
            "upgrade_priority": self.upgrade_priority,
            "is_base_platform": self.is_base_platform,
        }


class CompatibilityAnalyzer:
    """Determines platform compatibility for airsoft products."""

    def __init__(
        self,
        llm_client: OpenAI,
        llm_model: str = "local",
        # Callable searchers return List[Dict] with "text", "title" keys
        catalog_searcher: Optional[Callable] = None,
        vault_searcher: Optional[Callable] = None,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.catalog_searcher = catalog_searcher  # catalog_indexer.search_pages
        self.vault_searcher = vault_searcher       # vault_indexer.search_recommendations

    # ── Keyword Detection (fast path) ──────────────────────────────

    def keyword_detect(self, product: Dict[str, Any]) -> CompatibilityResult:
        """Fast keyword-based detection. Handles obvious cases without LLM."""
        title = product.get("title", "")
        handle = product.get("handle", "")
        tags = product.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        brand = product.get("brand", "") or product.get("vendor", "")
        product_type = product.get("product_type", "")
        description = product.get("text", "") or product.get("body_html", "")

        text = f"{title} {handle} {' '.join(tags)} {brand} {description}".lower()

        # 1. Check if cross-platform category
        category = product_type.lower().strip() if product_type else ""
        tag_set = {t.lower().strip() for t in tags}
        if category in CROSS_PLATFORM_CATEGORIES or tag_set & CROSS_PLATFORM_CATEGORIES:
            return CompatibilityResult(
                confidence="high",
                reasoning=f"Cross-platform category: {category or tag_set & CROSS_PLATFORM_CATEGORIES}",
            )

        # 2. Check if this IS a base platform (gun/replica)
        is_gun = any(indicator in text for indicator in BASE_PLATFORM_INDICATORS)
        if is_gun:
            matched_platforms = self._match_platforms(text)
            return CompatibilityResult(
                compatible_platforms=matched_platforms[:1],  # Own platform slug
                is_base_platform=True,
                confidence="high" if matched_platforms else "medium",
                reasoning=f"Base platform detected: {matched_platforms}",
            )

        # 3. Detect upgrade type from title/tags
        upgrade_type = self._detect_upgrade_type(text)

        # 4. Detect compatible platforms from text
        matched_platforms = self._match_platforms(text)

        # 5. Barrel length cross-check for inner barrels
        if upgrade_type == "inner-barrel":
            barrel_length = self._extract_barrel_length(text)
            if barrel_length and matched_platforms:
                matched_platforms = self._validate_barrel_compatibility(
                    barrel_length, matched_platforms
                )

        confidence = "high" if matched_platforms and upgrade_type else "medium" if matched_platforms or upgrade_type else "low"

        return CompatibilityResult(
            compatible_platforms=matched_platforms,
            upgrade_type=upgrade_type,
            upgrade_priority=self._infer_priority(upgrade_type),
            confidence=confidence,
            reasoning=f"Keyword match: platforms={matched_platforms}, type={upgrade_type}",
        )

    def _match_platforms(self, text: str) -> List[str]:
        """Match platform slugs from text using keyword registry."""
        matched = []
        for slug, info in PLATFORMS.items():
            for kw in info["keywords"]:
                # Use word boundary for short keywords to avoid false positives
                if len(kw) <= 3:
                    if re.search(rf'\b{re.escape(kw)}\b', text):
                        matched.append(slug)
                        break
                else:
                    if kw in text:
                        matched.append(slug)
                        break
        return matched

    def _detect_upgrade_type(self, text: str) -> Optional[str]:
        """Detect upgrade type from product text."""
        type_keywords = {
            "inner-barrel": ["inner barrel", "tight bore", "precision barrel", "6.01", "6.03", "6.04", "6.05", "crazy jet", "morpheus"],
            "outer-barrel": ["outer barrel", "threaded barrel", "barrel extension"],
            "hop-up": ["hop-up", "hop up", "bucking", "nub", "tensioner", "macaron", "autobot", "decepticon"],
            "spring": [" m90", " m100", " m110", " m120", " m130", " m140", " m150", " m160", " m170", " m180", " m190", "joule creep", "main spring"],
            "trigger": ["trigger unit", "speed trigger", "cnc trigger"],
            "cylinder": ["cylinder set", "cylinder head", "air nozzle"],
            "piston": ["piston head", "full tooth piston"],
            "gearbox": ["gearbox", "gear set", "gears 13:1", "gears 16:1", "gears 18:1", " etu "],
            "motor": ["high torque motor", "high speed motor"],
            "body-kit": ["folding stock", "pistol grip", "handguard", "body kit", "rail system", "m-lok", "keymod"],
            "rail-mount": ["scope mount", "riser mount", "picatinny rail"],
            "suppressor": ["suppressor", "silencer", "flash hider", "tracer unit"],
            "magazine": ["magazine", "midcap", "hicap", "speed loader", "gas magazine"],
            "gas-system": ["gas valve", "npas", "gas router", "high flow valve"],
            "optic": ["scope", "red dot", "holographic sight", "magnifier"],
            "bb": [" bbs ", "0.20g", "0.25g", "0.28g", "0.30g", "0.32g", "0.36g", "0.40g", "0.43g", "0.45g"],
            "battery": ["lipo battery", "nimh battery", "smart charger"],
            "tool": ["allen key set", "silicone oil", "maintenance kit"],
            "upgrade-kit": ["upgrade kit", "tuning kit", "full upgrade", "conversion kit"],
        }
        for type_slug, keywords in type_keywords.items():
            for kw in keywords:
                if kw in text:
                    return type_slug
        return None

    def _extract_barrel_length(self, text: str) -> Optional[int]:
        """Extract barrel length in mm from text."""
        match = re.search(r'(\d{2,3})\s*mm', text)
        if match:
            length = int(match.group(1))
            if 50 <= length <= 700:  # Reasonable airsoft barrel range
                return length
        return None

    def _validate_barrel_compatibility(
        self, barrel_length: int, platforms: List[str], tolerance: int = 15
    ) -> List[str]:
        """Validate barrel length against platform specs. Remove incompatible platforms."""
        valid = []
        for slug in platforms:
            platform = PLATFORMS.get(slug, {})
            lengths = platform.get("barrel_lengths", [])
            if not lengths:
                valid.append(slug)  # No length data, keep it
                continue
            if any(abs(barrel_length - pl) <= tolerance for pl in lengths):
                valid.append(slug)
        return valid

    def _infer_priority(self, upgrade_type: Optional[str]) -> Optional[str]:
        """Infer upgrade priority from type."""
        if not upgrade_type:
            return None
        essential = {"inner-barrel", "hop-up", "spring", "cylinder"}
        recommended = {"trigger", "piston", "gearbox", "motor", "gas-system"}
        if upgrade_type in essential:
            return "essential"
        if upgrade_type in recommended:
            return "recommended"
        return "optional"

    # ── LLM Analysis (slow path for ambiguous products) ────────────

    async def gather_guide_context(self, candidate_platforms: List[str]) -> str:
        """Fetch guide pages + recommendation notes for candidate platforms.

        Note: catalog_searcher and vault_searcher return List[Dict] with keys
        like "text", "title", etc. — NOT Qdrant ScoredPoint objects.
        """
        context_parts = []

        for slug in candidate_platforms[:3]:  # Max 3 platforms
            platform = PLATFORMS.get(slug, {})
            platform_name = platform.get("name", slug)

            # Search guide pages (returns List[Dict])
            if self.catalog_searcher:
                try:
                    guide_results = self.catalog_searcher(
                        query=f"{platform_name} upgrade guide compatible parts",
                        top_k=3,
                    )
                    for r in guide_results:
                        text = r.get("text", "")[:800]
                        if text:
                            context_parts.append(f"[Guide: {platform_name}] {text}")
                except Exception as e:
                    logger.warning(f"Failed to fetch guide context for {slug}: {e}")

            # Search vault recommendations (returns List[Dict])
            if self.vault_searcher:
                try:
                    vault_results = self.vault_searcher(
                        query=f"{platform_name} upgrades recommendations",
                        top_k=2,
                    )
                    for r in vault_results:
                        text = r.get("text", "")[:800]
                        if text:
                            context_parts.append(f"[Expert Rec: {platform_name}] {text}")
                except Exception as e:
                    logger.warning(f"Failed to fetch vault context for {slug}: {e}")

        return "\n\n".join(context_parts[:8])  # Max ~6400 chars

    async def llm_analyze(
        self, product: Dict[str, Any], guide_context: str
    ) -> CompatibilityResult:
        """LLM determines compatibility with structured JSON output."""
        title = product.get("title", "")
        description = (product.get("text", "") or product.get("body_html", ""))[:500]
        brand = product.get("brand", "") or product.get("vendor", "")
        tags = product.get("tags", [])
        if isinstance(tags, list):
            tags = ", ".join(tags)

        platform_summary = "\n".join(
            f"- {slug}: {info['name']} ({info['type']}) — barrel lengths: {info['barrel_lengths']}mm"
            for slug, info in PLATFORMS.items()
        )

        upgrade_types_summary = "\n".join(
            f"- {slug}: {desc}" for slug, desc in UPGRADE_TYPES.items()
        )

        prompt = f"""You are an airsoft technician analyzing product compatibility.

PRODUCT:
- Title: {title}
- Description: {description}
- Brand: {brand}
- Tags: {tags}

AVAILABLE PLATFORMS:
{platform_summary}

UPGRADE TYPES:
{upgrade_types_summary}

EXPERT GUIDE CONTEXT:
{guide_context if guide_context else "No guide context available."}

Return ONLY valid JSON (no markdown, no code fences):
{{
  "compatible_platforms": ["platform-slug"],
  "upgrade_type": "type-slug-or-null",
  "upgrade_priority": "essential-or-recommended-or-optional-or-null",
  "is_base_platform": false,
  "reasoning": "brief explanation"
}}

RULES:
1. Only assign platforms if CONFIDENT about compatibility
2. Product title mentioning a specific model = strong signal
3. Barrel length is key for inner barrels (check against platform specs)
4. Universal/cross-platform items → empty compatible_platforms
5. BBs, gas, tools, generic gear → empty compatible_platforms
6. If product IS a gun/replica → is_base_platform=true, compatible_platforms=[own-slug]
7. Use ONLY slugs from the AVAILABLE PLATFORMS list
8. Use ONLY slugs from the UPGRADE TYPES list"""

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    timeout=60,
                    extra_body={"request_timeout": 60},
                ),
            )
            content = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)

            data = json.loads(content)

            # Validate slugs against registries
            valid_platforms = [p for p in data.get("compatible_platforms", []) if p in PLATFORMS]
            valid_type = data.get("upgrade_type") if data.get("upgrade_type") in UPGRADE_TYPES else None
            valid_priority = data.get("upgrade_priority") if data.get("upgrade_priority") in ("essential", "recommended", "optional") else None

            return CompatibilityResult(
                compatible_platforms=valid_platforms,
                upgrade_type=valid_type,
                upgrade_priority=valid_priority or self._infer_priority(valid_type),
                is_base_platform=bool(data.get("is_base_platform", False)),
                confidence="medium",
                reasoning=data.get("reasoning", "LLM analysis"),
            )
        except Exception as e:
            logger.error(f"LLM compatibility analysis failed for '{title}': {e}")
            return CompatibilityResult(confidence="low", reasoning=f"LLM error: {e}")

    # ── Main Entry Point ───────────────────────────────────────────

    async def analyze_product(self, product: Dict[str, Any]) -> CompatibilityResult:
        """Analyze a single product. Keyword fast path, LLM fallback."""
        result = self.keyword_detect(product)

        if result.confidence == "high":
            return result

        # Low/medium confidence → use LLM with guide context
        candidate_platforms = result.compatible_platforms or list(PLATFORMS.keys())[:5]
        guide_context = await self.gather_guide_context(candidate_platforms)
        llm_result = await self.llm_analyze(product, guide_context)

        # Prefer LLM result but keep keyword data as fallback
        if llm_result.confidence != "low":
            return llm_result

        return result  # Fall back to keyword result
