"""
Knowledge Synthesizer — Generates product recommendation notes from
Shopify guides, catalog, competitor data, and community sources.

Runs after vault_builder + content_learner in the daily rebuild cycle.
Produces goal-oriented recommendation notes in the Obsidian vault.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import jinja2

logger = logging.getLogger(__name__)

# ── Platform Registry (V1: Sniper + GBB) ────────────────────────────

PLATFORMS = {
    "srs-a2": {
        "name": "Silverback SRS A2/M2",
        "type": "sniper",
        "keywords": ["srs", "srs a2", "srs m2", "srs-a2"],
        "brands": ["Silverback", "STALKER"],
        "goals": ["accuracy", "power", "silence", "reliability", "bb-selection"],
    },
    "tac-41": {
        "name": "Silverback TAC-41",
        "type": "sniper",
        "keywords": ["tac-41", "tac41", "tac 41"],
        "brands": ["Silverback", "STALKER"],
        "goals": ["accuracy", "power", "silence", "reliability", "bb-selection"],
    },
    "vsr-10": {
        "name": "VSR-10 Platform",
        "type": "sniper",
        "keywords": ["vsr", "vsr-10", "vsr10"],
        "brands": ["Tokyo Marui", "Action Army", "Maple Leaf"],
        "goals": ["accuracy", "power", "bb-selection", "budget-build"],
    },
    "glock": {
        "name": "Tokyo Marui Glock",
        "type": "gbb",
        "keywords": ["glock", "g17", "g18", "g19", "g-series", "g series"],
        "brands": ["Tokyo Marui", "Wii Tech", "Maple Leaf"],
        "goals": ["accuracy", "reliability", "bb-selection"],
    },
    "mws": {
        "name": "Tokyo Marui MWS M4 GBBR",
        "type": "gbb",
        "keywords": ["mws", "m4 gbb", "m4 gbbr"],
        "brands": ["Tokyo Marui", "Wii Tech"],
        "goals": ["accuracy", "power", "reliability", "bb-selection"],
    },
}

# Categories that are cross-platform (no platform-specific node)
CROSS_PLATFORM_CATEGORIES = {"ammunition", "optic", "gear", "protection"}

# Cross-platform guide notes to generate
CROSS_PLATFORM_GUIDES = [
    {"slug": "bb-selection-sniper", "title": "BB Selection for Sniper Platforms",
     "goal": "bb-selection", "platforms": ["srs-a2", "tac-41", "vsr-10"]},
    {"slug": "bb-selection-gbb", "title": "BB Selection for GBB Platforms",
     "goal": "bb-selection", "platforms": ["glock", "mws"]},
    {"slug": "spring-selection-guide", "title": "Spring Selection Guide (All Snipers)",
     "goal": "power", "platforms": ["srs-a2", "tac-41", "vsr-10"]},
    {"slug": "gbb-gas-and-maintenance", "title": "GBB Gas Types & Maintenance",
     "goal": "reliability", "platforms": ["glock", "mws"]},
    {"slug": "gbb-magazine-guide", "title": "GBB Magazine Guide",
     "goal": "reliability", "platforms": ["glock", "mws"]},
]

HASH_PREFIX = "synth:hash"


class KnowledgeSynthesizer:
    """Generates product recommendation notes from multiple knowledge sources."""

    def __init__(
        self,
        vault_path: str,
        llm_client,
        product_indexer,
        catalog_indexer=None,
        vault_indexer=None,
        redis_client=None,
    ):
        self.vault_path = Path(vault_path)
        self.rec_path = self.vault_path / "recommendations"
        self.llm_client = llm_client
        self.product_indexer = product_indexer
        self.catalog_indexer = catalog_indexer
        self.vault_indexer = vault_indexer
        self.redis = redis_client
        self.llm_model = os.getenv("LLM_MODEL", "local")

        # Jinja2 environment (same pattern as vault_builder)
        template_dir = self.vault_path / "_templates"
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            undefined=jinja2.StrictUndefined,
        )

    # ── Phase 1: GATHER ──────────────────────────────────────────

    def _detect_platform(self, product: dict) -> List[str]:
        """Determine which platform(s) a product belongs to via keyword matching."""
        text = f"{product.get('title', '')} {' '.join(product.get('tags', []))} {product.get('brand', '')}".lower()

        if product.get("category") in CROSS_PLATFORM_CATEGORIES:
            return []  # Cross-platform, handled by guide notes

        matched = []
        for slug, info in PLATFORMS.items():
            for kw in info["keywords"]:
                if kw in text:
                    matched.append(slug)
                    break

        # Brand fallback: Silverback → sniper, Tokyo Marui → gbb
        if not matched:
            brand = product.get("brand", "").lower()
            if brand in ("silverback", "stalker"):
                matched = ["srs-a2"]  # Default sniper platform
            elif brand == "tokyo marui":
                matched = ["glock"]  # Default GBB platform

        return matched

    async def _gather_guide_content(self, platform_slug: str) -> List[dict]:
        """Fetch Shopify guide chunks relevant to a platform."""
        if not self.catalog_indexer:
            return []
        platform = PLATFORMS.get(platform_slug, {})
        query = f"{platform.get('name', '')} guide upgrade"
        try:
            return self.catalog_indexer.search_pages(query, top_k=20)
        except Exception as e:
            logger.warning(f"Guide fetch failed for {platform_slug}: {e}")
            return []

    async def _gather_products_for_platform(self, platform_slug: str) -> dict:
        """Fetch products matching a platform, split by stock status."""
        platform = PLATFORMS.get(platform_slug, {})
        keywords = platform.get("keywords", [])
        brands = platform.get("brands", [])

        all_products = []
        # Search by each brand associated with platform
        for brand in brands:
            results = self.product_indexer.search(
                query=f"{platform.get('name', '')} {brand}",
                top_k=30,
                brand_filter=brand,
            )
            all_products.extend(results)

        # Also search by keywords
        for kw in keywords[:2]:
            results = self.product_indexer.search(query=kw, top_k=20)
            all_products.extend(results)

        # Deduplicate by handle
        seen = set()
        unique = []
        for p in all_products:
            h = p.get("handle", "")
            if h and h not in seen:
                seen.add(h)
                unique.append(p)

        in_stock = [p for p in unique if (p.get("inventory_quantity") or 0) > 0]
        out_of_stock = [p for p in unique if (p.get("inventory_quantity") or 0) <= 0]

        return {"in_stock": in_stock, "out_of_stock": out_of_stock}

    async def _gather_competitor_intel(self, platform_slug: str) -> List[dict]:
        """Fetch competitor notes relevant to a platform."""
        if not self.vault_indexer:
            return []
        platform = PLATFORMS.get(platform_slug, {})
        query = f"{platform.get('name', '')} products pricing"
        try:
            return self.vault_indexer.search_recommendations(
                query=query, top_k=5, note_type="competitor"
            )
        except Exception:
            return []

    async def _gather_context(self, platform_slug: str, goal: str) -> dict:
        """Assemble full context package for a platform x goal."""
        guide_chunks = await self._gather_guide_content(platform_slug)
        products = await self._gather_products_for_platform(platform_slug)
        competitor_intel = await self._gather_competitor_intel(platform_slug)

        return {
            "platform": PLATFORMS[platform_slug],
            "platform_slug": platform_slug,
            "goal": goal,
            "guide_chunks": guide_chunks,
            "products_in_stock": products["in_stock"],
            "products_out_of_stock": products["out_of_stock"],
            "competitor_intel": competitor_intel,
        }

    # ── Phase 2: BUILD PRODUCT GRAPH ─────────────────────────────

    def _build_product_list_text(self, products: List[dict], label: str) -> str:
        """Format product list for LLM prompt."""
        if not products:
            return f"{label}: (none available)"
        lines = [f"{label}:"]
        for p in products[:20]:  # Cap at 20 to fit context
            stock = p.get("inventory_quantity", 0)
            stock_label = f"stock: {stock}" if stock > 0 else "OUT OF STOCK"
            lines.append(
                f"- {p.get('title', '?')} | handle: {p.get('handle', '?')} "
                f"| €{p.get('price', '?')} | {p.get('brand', '?')} | {stock_label}"
            )
        return "\n".join(lines)

    def _build_guide_context(self, guide_chunks: List[dict], max_chars: int = 3000) -> str:
        """Format guide content for LLM prompt."""
        if not guide_chunks:
            return "No guide content available for this platform."
        text_parts = []
        total = 0
        for chunk in guide_chunks:
            section = chunk.get("section_heading", "")
            content = chunk.get("text", "")
            entry = f"### {section}\n{content}" if section else content
            if total + len(entry) > max_chars:
                break
            text_parts.append(entry)
            total += len(entry)
        return "\n\n".join(text_parts) if text_parts else "No guide content available."

    # ── Phase 3: SYNTHESIZE ──────────────────────────────────────

    async def _synthesize_note(self, context: dict) -> Optional[str]:
        """Generate a recommendation note via LLM."""
        platform = context["platform"]
        goal = context["goal"]

        guide_text = self._build_guide_context(context["guide_chunks"])
        in_stock_text = self._build_product_list_text(
            context["products_in_stock"], "OUR PRODUCTS (in stock)"
        )
        oos_text = self._build_product_list_text(
            context["products_out_of_stock"], "OUR PRODUCTS (out of stock — mention as alternatives)"
        )

        # Collect all provided handles for validation
        all_handles = set()
        for p in context["products_in_stock"] + context["products_out_of_stock"]:
            h = p.get("handle", "")
            if h:
                all_handles.add(h)

        prompt = f"""You are a senior airsoft technician at Skirmshop.es writing expert product recommendations.

TASK: Write a {goal} recommendation note for the {platform['name']} platform.

CONTEXT FROM OUR GUIDES:
{guide_text}

{in_stock_text}

{oos_text}

RULES:
1. ONLY recommend products from the lists above — never invent handles
2. For EVERY product you recommend, include the exact tool call with the handle from the product list. Example: [TOOL:show_product:stalker-kraken-srs-70-concave-bucking]
3. Use the EXACT handle string from the product lists above — do not modify, abbreviate, or wrap in braces
4. Be specific: FPS ranges, spring weights, barrel specs, joule ratings
5. If a critical part is out of stock, say so clearly and suggest alternatives from the in-stock list
6. Structure as: Goal explanation → Priority-ordered upgrade steps → Compatibility notes → Budget tiers
7. Each upgrade step: name the part, explain WHY, then the product tool call
8. Write in English (the chatbot translates for the customer)
9. Maximum 1500 words"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    timeout=180,
                    extra_body={"request_timeout": 180},  # LiteLLM proxy timeout
                ),
            )
            content = response.choices[0].message.content

            # Strip thinking tokens (same pattern as vault_builder)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            content = re.sub(r"```markdown\s*", "", content)
            content = re.sub(r"```\s*$", "", content)

            # Validate handles
            content = self._validate_generated_content(content, all_handles)

            return content
        except Exception as e:
            logger.error(f"LLM synthesis failed for {platform['name']}/{goal}: {e}")
            return None

    def _validate_generated_content(self, content: str, provided_handles: set) -> str:
        """Strip hallucinated product handles from generated content."""
        generated_handles = re.findall(r'\[TOOL:show_product:([^\]]+)\]', content)

        invalid_count = 0
        for handle in generated_handles:
            if handle not in provided_handles:
                content = content.replace(f'[TOOL:show_product:{handle}]', '')
                logger.warning(f"Stripped hallucinated handle: {handle}")
                invalid_count += 1

        total = len(generated_handles)
        if total > 0 and invalid_count / total > 0.5:
            logger.error(
                f"Rejected generation: {invalid_count}/{total} handles were invalid"
            )
            return ""  # Empty string is falsy — caller's `if not llm_content:` catches it

        return content

    # ── Phase 4: RENDER & WRITE ──────────────────────────────────

    def _preserve_human_section(self, note_path: Path) -> str:
        """Extract human-preserved section from existing note."""
        if not note_path.exists():
            return ""
        existing = note_path.read_text()
        match = re.search(
            r"<!-- human-start -->(.*?)<!-- human-end -->",
            existing, re.DOTALL,
        )
        return match.group(1) if match else ""

    def _render_template(self, template_name: str, context: dict) -> str:
        """Render a Jinja2 template."""
        template = self.jinja_env.get_template(template_name)
        return template.render(**context)

    def _extract_referenced_handles(self, content: str) -> List[str]:
        """Extract all product handles from generated content."""
        return re.findall(r'\[TOOL:show_product:([^\]]+)\]', content)

    async def _write_recommendation_note(
        self,
        subdir: str,
        filename: str,
        template_name: str,
        template_context: dict,
        llm_content: str,
    ):
        """Write a recommendation note, preserving human sections."""
        out_dir = self.rec_path / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        note_path = out_dir / filename

        # Preserve human edits
        human_block = self._preserve_human_section(note_path)

        # Add LLM content and referenced handles to template context
        template_context["llm_content"] = llm_content
        template_context["products_referenced"] = self._extract_referenced_handles(llm_content)
        template_context["last_synthesized"] = datetime.now(timezone.utc).isoformat()

        rendered = self._render_template(template_name, template_context)

        # Splice human block back in
        if human_block:
            default_human = (
                "<!-- human-start -->\n"
                "## Expert Notes\n"
                "_Your manual additions preserved across rebuilds._\n"
                "<!-- human-end -->"
            )
            rendered = rendered.replace(
                default_human,
                f"<!-- human-start -->{human_block}<!-- human-end -->",
            )

        note_path.write_text(rendered)
        logger.info(f"Wrote recommendation: {note_path.relative_to(self.vault_path)}")

    # ── Content Hash (skip unchanged) ────────────────────────────

    async def _compute_hash(self, key: str, content: str) -> str:
        return hashlib.sha256(f"{key}:{content}".encode()).hexdigest()

    async def _has_changed(self, key: str, content: str) -> bool:
        if not self.redis:
            return True
        new_hash = await self._compute_hash(key, content)
        redis_key = f"{HASH_PREFIX}:{key}"
        old_hash = await self.redis.get(redis_key)
        if old_hash:
            old_hash = old_hash.decode() if isinstance(old_hash, bytes) else old_hash
        return old_hash != new_hash

    async def _set_hash(self, key: str, content: str):
        if not self.redis:
            return
        h = await self._compute_hash(key, content)
        await self.redis.set(f"{HASH_PREFIX}:{key}", h)

    # ── Phase 5: MAIN ORCHESTRATOR ───────────────────────────────

    async def synthesize(self, force: bool = False) -> dict:
        """Main entry point. Generates all recommendation notes."""
        stats = {
            "notes_generated": 0,
            "notes_skipped": 0,
            "notes_failed": 0,
            "llm_calls": 0,
            "handles_validated": 0,
            "platforms": {},
        }

        # 1. Generate platform x goal notes
        for platform_slug, platform_info in PLATFORMS.items():
            platform_count = 0
            for goal in platform_info["goals"]:
                note_key = f"{platform_slug}/{goal}"

                try:
                    # Gather context
                    context = await self._gather_context(platform_slug, goal)

                    # Build a hash key from the context to detect changes
                    context_summary = json.dumps({
                        "guides": len(context["guide_chunks"]),
                        "in_stock": [p["handle"] for p in context["products_in_stock"][:20]],
                        "oos": [p["handle"] for p in context["products_out_of_stock"][:10]],
                    }, sort_keys=True)

                    if not force and not await self._has_changed(note_key, context_summary):
                        stats["notes_skipped"] += 1
                        continue

                    # Synthesize via LLM
                    llm_content = await self._synthesize_note(context)
                    stats["llm_calls"] += 1

                    if not llm_content:
                        stats["notes_failed"] += 1
                        continue

                    # Write the note
                    filename = f"{platform_slug}-{goal}.md"
                    await self._write_recommendation_note(
                        subdir="products",
                        filename=filename,
                        template_name="recommendation.md.j2",
                        template_context={
                            "handle": f"{platform_slug}-{goal}",
                            "title": f"{platform_info['name']} — {goal.replace('-', ' ').title()}",
                            "platform": platform_slug,
                            "category": platform_info["type"],
                            "price": "",
                            "source_guides": [f"{platform_info['name']} Guide"],
                            "confidence": "high" if context["guide_chunks"] else "medium",
                        },
                        llm_content=llm_content,
                    )

                    await self._set_hash(note_key, context_summary)
                    stats["notes_generated"] += 1
                    platform_count += 1

                except Exception as e:
                    logger.error(f"Failed to synthesize {note_key}: {e}")
                    stats["notes_failed"] += 1

            stats["platforms"][platform_slug] = platform_count

        # 2. Generate cross-platform guide notes
        for guide in CROSS_PLATFORM_GUIDES:
            note_key = f"guide/{guide['slug']}"
            try:
                # Gather context from all related platforms
                all_guide_chunks = []
                all_products_in_stock = []
                all_products_oos = []
                for ps in guide["platforms"]:
                    ctx = await self._gather_context(ps, guide["goal"])
                    all_guide_chunks.extend(ctx["guide_chunks"])
                    all_products_in_stock.extend(ctx["products_in_stock"])
                    all_products_oos.extend(ctx["products_out_of_stock"])

                # Deduplicate products by handle
                seen = set()
                deduped_is = []
                for p in all_products_in_stock:
                    if p["handle"] not in seen:
                        seen.add(p["handle"])
                        deduped_is.append(p)
                deduped_oos = []
                for p in all_products_oos:
                    if p["handle"] not in seen:
                        seen.add(p["handle"])
                        deduped_oos.append(p)

                merged_context = {
                    "platform": {"name": guide["title"]},
                    "platform_slug": "cross-platform",
                    "goal": guide["goal"],
                    "guide_chunks": all_guide_chunks,
                    "products_in_stock": deduped_is,
                    "products_out_of_stock": deduped_oos,
                    "competitor_intel": [],
                }

                context_summary = json.dumps({
                    "guides": len(all_guide_chunks),
                    "in_stock": [p["handle"] for p in deduped_is[:20]],
                }, sort_keys=True)

                if not force and not await self._has_changed(note_key, context_summary):
                    stats["notes_skipped"] += 1
                    continue

                llm_content = await self._synthesize_note(merged_context)
                stats["llm_calls"] += 1

                if not llm_content:
                    stats["notes_failed"] += 1
                    continue

                await self._write_recommendation_note(
                    subdir="guides",
                    filename=f"{guide['slug']}.md",
                    template_name="recommendation-guide.md.j2",
                    template_context={
                        "title": guide["title"],
                        "slug": guide["slug"],
                        "platforms": guide["platforms"],
                        "goal": guide["goal"],
                        "source_guides": [PLATFORMS[p]["name"] for p in guide["platforms"]],
                        "confidence": "high" if all_guide_chunks else "medium",
                    },
                    llm_content=llm_content,
                )

                await self._set_hash(note_key, context_summary)
                stats["notes_generated"] += 1

            except Exception as e:
                logger.error(f"Failed to synthesize guide {guide['slug']}: {e}")
                stats["notes_failed"] += 1

        # 3. Write stats
        meta_path = self.vault_path / "_meta" / "last-synthesis.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        stats["last_run"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(stats, indent=2))

        logger.info(f"Synthesis complete: {stats}")
        return stats
