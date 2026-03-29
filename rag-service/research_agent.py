"""
Research Agent — Iterative intelligence loop that continuously improves
competitor analysis by researching gaps via web search.

Runs hourly. Each iteration:
1. Reads existing vault notes, identifies gaps (missing brands, categories, prices)
2. Searches Google/web for specific intelligence to fill those gaps
3. Uses LLM to merge new findings with existing knowledge
4. Updates notes with richer data

For CF-blocked sites where we can't crawl, this is the PRIMARY intelligence source.
"""

import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from firecrawl_client import FirecrawlClient

logger = logging.getLogger(__name__)

# ── Research Skills: what to learn for each competitor ──
RESEARCH_SKILLS = {
    "brands": {
        "description": "Discover all airsoft brands carried by this competitor",
        "queries": [
            "{name} marcas airsoft",
            "{name} brands airsoft Spain",
            "site:{domain} tokyo marui OR vfc OR g&g OR cyma OR specna",
            "{name} distribuidor marcas",
        ],
        "extract_prompt": (
            "From these search results about '{name}', extract ALL airsoft brand names mentioned. "
            "Return as JSON: {{\"brands\": [{{\"name\": \"Brand Name\", \"evidence\": \"brief source\"}}]}}"
        ),
    },
    "categories": {
        "description": "Understand product categories and catalog depth",
        "queries": [
            "site:{domain} réplicas OR pistolas OR rifles OR accesorios",
            "{name} catálogo airsoft productos",
            "{name} tienda categorías",
        ],
        "extract_prompt": (
            "From these search results about '{name}', identify their product categories. "
            "Return as JSON: {{\"categories\": [{{\"name\": \"Category\", \"estimated_depth\": \"deep/medium/thin\", \"evidence\": \"brief\"}}]}}"
        ),
    },
    "prices": {
        "description": "Understand pricing strategy and price ranges",
        "queries": [
            "{name} precios airsoft",
            "site:{domain} €",
            "{name} ofertas descuentos airsoft",
            "{name} review precio opiniones",
        ],
        "extract_prompt": (
            "From these search results about '{name}', extract pricing intelligence. "
            "Return as JSON: {{\"price_range\": \"€X - €Y\", \"strategy\": \"budget/mid/premium\", "
            "\"notable_prices\": [{{\"product\": \"name\", \"price\": \"€X\"}}], \"discounting\": \"description\"}}"
        ),
    },
    "services": {
        "description": "Discover services beyond retail (workshop, repairs, events)",
        "queries": [
            "{name} taller reparaciones airsoft",
            "{name} servicios tienda",
            "{name} workshop custom upgrade",
            "{name} eventos partidas airsoft",
        ],
        "extract_prompt": (
            "From these search results about '{name}', identify services they offer beyond selling products. "
            "Return as JSON: {{\"services\": [{{\"name\": \"Service\", \"description\": \"brief\", \"evidence\": \"source\"}}]}}"
        ),
    },
    "reputation": {
        "description": "Community reputation, reviews, recommendations",
        "queries": [
            "{name} opiniones review airsoft",
            "{name} experiencia compra",
            "reddit {name} airsoft",
            "{name} trustpilot OR google reviews",
            "mejor tienda airsoft españa {name}",
        ],
        "extract_prompt": (
            "From these search results, summarize the community reputation of '{name}'. "
            "Return as JSON: {{\"overall_sentiment\": \"positive/mixed/negative\", "
            "\"strengths_mentioned\": [\"list\"], \"complaints\": [\"list\"], "
            "\"recommendation_rate\": \"high/medium/low\", \"evidence\": \"brief summary\"}}"
        ),
    },
    "platform": {
        "description": "Technical platform, shipping, payment methods",
        "queries": [
            "site:{domain} envío OR shipping OR pago",
            "{name} tienda online plataforma",
            "{domain} wappalyzer OR builtwith",
        ],
        "extract_prompt": (
            "From these search results about '{name}', identify their e-commerce platform and logistics. "
            "Return as JSON: {{\"platform\": \"PrestaShop/WooCommerce/Shopify/custom\", "
            "\"shipping\": \"description\", \"payment_methods\": [\"list\"], \"physical_store\": \"yes/no/unknown\", "
            "\"location\": \"city if known\"}}"
        ),
    },
}


class ResearchAgent:
    """Iterative research agent that fills intelligence gaps via web search."""

    def __init__(self, firecrawl: FirecrawlClient, llm_client, vault_path: str):
        self.fc = firecrawl
        self.llm = llm_client
        self.vault_path = Path(vault_path)
        self.model = os.getenv("LLM_MODEL", "local")

    async def research_all(self, competitors: List[dict] = None) -> dict:
        """Research all competitors, filling gaps in existing analyses."""
        result = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "competitors_researched": 0,
            "gaps_filled": 0,
            "notes_updated": 0,
            "errors": [],
        }

        targets = competitors or self._load_competitors_from_vault()
        if not targets:
            result["errors"].append("No competitors to research")
            return result

        # Global timeout: 8 minutes max to stay within cron's 10-minute curl timeout
        MAX_RESEARCH_SECONDS = 480

        import time
        deadline = time.monotonic() + MAX_RESEARCH_SECONDS

        for target in targets:
            if time.monotonic() > deadline:
                logger.warning("Research hit global timeout, stopping")
                result["errors"].append(f"Global timeout ({MAX_RESEARCH_SECONDS}s) reached")
                break

            try:
                gaps = self._identify_gaps(target)
                if not gaps:
                    logger.info(f"[{target['name']}] No gaps to fill, skipping")
                    continue

                logger.info(f"[{target['name']}] Found {len(gaps)} gaps: {', '.join(gaps)}")
                findings = await self._research_gaps(target, gaps)

                if findings:
                    updated = self._merge_findings(target, findings)
                    if updated:
                        result["notes_updated"] += 1
                    result["gaps_filled"] += len(findings)

                result["competitors_researched"] += 1

            except Exception as e:
                logger.error(f"Research failed for {target.get('name', '?')}: {e}")
                result["errors"].append(f"{target.get('name', '?')}: {str(e)[:200]}")

        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        return result

    def _load_competitors_from_vault(self) -> List[dict]:
        """Load competitor info from existing deep analysis notes."""
        competitors = []
        comp_dir = self.vault_path / "competitors"
        if not comp_dir.exists():
            return []

        for f in comp_dir.glob("*-deep.md"):
            try:
                content = f.read_text(encoding="utf-8")
                # Parse YAML frontmatter
                if content.startswith("---"):
                    end = content.index("---", 3)
                    fm = content[3:end].strip()
                    meta = {}
                    for line in fm.split("\n"):
                        if ":" in line:
                            key, val = line.split(":", 1)
                            meta[key.strip()] = val.strip()
                    competitors.append({
                        "name": meta.get("name", f.stem),
                        "slug": meta.get("slug", f.stem.replace("-deep", "")),
                        "url": meta.get("url", ""),
                        "domain": meta.get("url", "").replace("https://", "").replace("http://", "").rstrip("/"),
                        "note_path": str(f),
                        "content": content,
                    })
            except Exception:
                pass

        return competitors

    def _identify_gaps(self, target: dict) -> List[str]:
        """Identify what intelligence is missing from the existing analysis."""
        content = target.get("content", "")
        content_lower = content.lower()
        gaps = []

        # Check each skill area for gaps
        gap_indicators = {
            "brands": [
                "no brands were identified",
                "no brand names visible",
                "brand portfolio\n_pending",
                "cannot identify",
                "## brand portfolio\nno brands",
            ],
            "categories": [
                "no category depth",
                "cannot identify product categories",
                "category breakdown\n_pending",
                "no product categories",
            ],
            "prices": [
                "impossible to analyze pricing",
                "no price points identified",
                "pricing strategy\n_pending",
                "no product prices",
            ],
            "services": [
                "no custom builds or event",
                "no unique differentiators",
                "services & differentiators\n_pending",
            ],
            "reputation": [
                # Always research reputation — it's never in the crawled data
                "strategic notes\n_add your own",
            ],
            "platform": [
                "website & technology\n_pending",
                "## website & technology\nthe",
            ],
        }

        for skill, indicators in gap_indicators.items():
            for indicator in indicators:
                if indicator in content_lower:
                    gaps.append(skill)
                    break

        # If the note has very few pages analyzed, research everything
        pages_match = re.search(r"pages_analyzed:\s*(\d+)", content)
        if pages_match and int(pages_match.group(1)) <= 2:
            gaps = list(RESEARCH_SKILLS.keys())

        return list(set(gaps))

    async def _research_gaps(self, target: dict, gaps: List[str]) -> Dict[str, Any]:
        """Research specific gaps using web search."""
        findings = {}

        for gap in gaps:
            skill = RESEARCH_SKILLS.get(gap)
            if not skill:
                continue

            logger.info(f"[{target['name']}] Researching: {gap}")

            # Search the web for each query
            search_results = []
            for query_template in skill["queries"][:3]:  # Limit to 3 queries per skill
                query = query_template.format(
                    name=target["name"],
                    domain=target.get("domain", ""),
                )
                results = await self._web_search(query)
                if results:
                    search_results.extend(results)

            if not search_results:
                logger.info(f"[{target['name']}] No search results for {gap}")
                continue

            # Send to LLM for extraction
            combined_results = "\n\n".join(
                f"Source: {r.get('title', '?')} ({r.get('url', '')})\n{r.get('content', '')[:500]}"
                for r in search_results[:10]
            )

            extract_prompt = skill["extract_prompt"].format(name=target["name"])
            extracted = await self._llm_extract(extract_prompt, combined_results)

            if extracted:
                findings[gap] = extracted
                logger.info(f"[{target['name']}] Found {gap} data: {json.dumps(extracted)[:200]}")

        return findings

    _consecutive_search_failures: int = 0

    async def _web_search(self, query: str) -> List[dict]:
        # Fast-fail: if Firecrawl search consistently returns nothing, skip
        if self._consecutive_search_failures >= 3:
            logger.debug(f"Skipping search (Firecrawl consistently empty): {query}")
            return []
        """Search the web using Firecrawl's search endpoint."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    f"{self.fc.base_url}/v1/search",
                    json={"query": query, "limit": 5},
                    headers=self.fc._headers,
                )
                data = resp.json()
                if data.get("success") and data.get("data"):
                    return [
                        {
                            "title": r.get("metadata", {}).get("title", ""),
                            "url": r.get("metadata", {}).get("sourceURL", ""),
                            "content": r.get("markdown", "")[:800],
                        }
                        for r in data["data"]
                    ]
        except Exception as e:
            logger.debug(f"Firecrawl search failed for '{query}': {e}")

        self._consecutive_search_failures += 1
        # Google scrape fallback disabled — Google blocks Playwright scrapes,
        # causing cascading timeouts that hang the hourly cron job.
        logger.debug(f"No search results for '{query}', skipping (Google fallback disabled)")

        return []

    async def _llm_extract(self, prompt: str, search_data: str) -> Optional[dict]:
        """Use LLM to extract structured data from search results."""
        if not self.llm:
            return None

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": (
                            "You are a competitive intelligence researcher for Skirmshop.es (Spanish airsoft retailer). "
                            "Extract specific, factual intelligence from search results. "
                            "Return ONLY valid JSON. No thinking, no explanation."
                        )},
                        {"role": "user", "content": f"{prompt}\n\nSearch results:\n{search_data[:4000]}"},
                    ],
                    max_tokens=1500,
                    timeout=120,
                ),
            )
            text = response.choices[0].message.content.strip()
            # Strip thinking tokens
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            # Extract JSON
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"LLM extract failed: {e}")

        return None

    def _merge_findings(self, target: dict, findings: Dict[str, Any]) -> bool:
        """Merge research findings into the existing deep analysis note."""
        note_path = Path(target.get("note_path", ""))
        if not note_path.exists():
            return False

        content = note_path.read_text(encoding="utf-8")
        updated = False

        # Build research section
        research_lines = [
            f"\n## Research Findings (auto-updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC)\n"
        ]

        if "brands" in findings:
            brands_data = findings["brands"]
            brands_list = brands_data.get("brands", [])
            if brands_list:
                research_lines.append("### Brands Discovered via Research")
                for b in brands_list:
                    name = b.get("name", "") if isinstance(b, dict) else str(b)
                    evidence = b.get("evidence", "") if isinstance(b, dict) else ""
                    research_lines.append(f"- **{name}** — {evidence}")
                research_lines.append("")
                updated = True

        if "categories" in findings:
            cats = findings["categories"].get("categories", [])
            if cats:
                research_lines.append("### Categories Discovered via Research")
                for c in cats:
                    name = c.get("name", "") if isinstance(c, dict) else str(c)
                    depth = c.get("estimated_depth", "") if isinstance(c, dict) else ""
                    research_lines.append(f"- **{name}** ({depth})")
                research_lines.append("")
                updated = True

        if "prices" in findings:
            price_data = findings["prices"]
            research_lines.append("### Pricing Intelligence via Research")
            if price_data.get("price_range"):
                research_lines.append(f"- Price range: {price_data['price_range']}")
            if price_data.get("strategy"):
                research_lines.append(f"- Strategy: {price_data['strategy']}")
            if price_data.get("notable_prices"):
                for p in price_data["notable_prices"][:5]:
                    product = p.get("product", "") if isinstance(p, dict) else str(p)
                    price = p.get("price", "") if isinstance(p, dict) else ""
                    research_lines.append(f"- {product}: {price}")
            research_lines.append("")
            updated = True

        if "services" in findings:
            services = findings["services"].get("services", [])
            if services:
                research_lines.append("### Services Discovered via Research")
                for s in services:
                    name = s.get("name", "") if isinstance(s, dict) else str(s)
                    desc = s.get("description", "") if isinstance(s, dict) else ""
                    research_lines.append(f"- **{name}**: {desc}")
                research_lines.append("")
                updated = True

        if "reputation" in findings:
            rep = findings["reputation"]
            research_lines.append("### Community Reputation via Research")
            research_lines.append(f"- Overall sentiment: **{rep.get('overall_sentiment', 'unknown')}**")
            if rep.get("strengths_mentioned"):
                research_lines.append(f"- Praised for: {', '.join(rep['strengths_mentioned'][:5])}")
            if rep.get("complaints"):
                research_lines.append(f"- Complaints: {', '.join(rep['complaints'][:5])}")
            if rep.get("evidence"):
                research_lines.append(f"- Summary: {rep['evidence']}")
            research_lines.append("")
            updated = True

        if "platform" in findings:
            plat = findings["platform"]
            research_lines.append("### Platform & Logistics via Research")
            for key in ["platform", "shipping", "physical_store", "location"]:
                if plat.get(key):
                    research_lines.append(f"- {key.replace('_', ' ').title()}: {plat[key]}")
            if plat.get("payment_methods"):
                research_lines.append(f"- Payment: {', '.join(plat['payment_methods'])}")
            research_lines.append("")
            updated = True

        if updated:
            research_block = "\n".join(research_lines)
            # Replace existing research section or insert before human-start
            if "## Research Findings" in content:
                content = re.sub(
                    r"## Research Findings.*?(?=<!-- human-start -->)",
                    research_block + "\n",
                    content,
                    flags=re.DOTALL,
                )
            else:
                content = content.replace(
                    "<!-- human-start -->",
                    research_block + "\n<!-- human-start -->",
                )

            note_path.write_text(content, encoding="utf-8")
            logger.info(f"[{target['name']}] Updated note with {len(findings)} research findings")

        return updated
