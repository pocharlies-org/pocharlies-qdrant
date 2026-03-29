"""
Content Learner — Crawls non-competitor content sources, summarizes with LLM,
extracts entities (brands, categories, topics), and generates interlinked vault notes.

Different from vault_builder's product extraction: this focuses on knowledge, guides,
community discussions, and market trends rather than structured product data.
"""

import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# Known airsoft brands for entity extraction
KNOWN_BRANDS = [
    "Tokyo Marui", "VFC", "G&G", "KWA", "KSC", "WE", "ASG", "CYMA", "Classic Army",
    "ICS", "Krytac", "Maple Leaf", "Prometheus", "PDI", "Lonex", "SHS", "ZCI",
    "Novritsch", "Silverback", "Action Army", "Modify", "Gate", "Perun", "Jefftron",
    "Ares", "Amoeba", "S&T", "Specna Arms", "Double Eagle", "JG", "A&K", "LCT",
    "E&L", "Real Sword", "Systema", "Polar Star", "Wolverine", "Redline",
    "Helikon", "Invader Gear", "Emerson", "TMC", "Warrior Assault", "Condor",
    "Nuprol", "ASG", "Maxx Model", "Retro Arms", "CNC Production",
    "Acetech", "Xcortech", "Evolution", "King Arms", "Umarex", "Elite Force",
    "Lancer Tactical", "Valken", "HFC", "KJW", "Marui", "TM", "Saigo",
    "Rossi", "EmersonGear", "Clawgear", "5.11", "Mechanix", "Oakley",
]

KNOWN_CATEGORIES = [
    "AEG", "GBB", "GBBR", "Sniper", "Pistol", "Shotgun", "SMG", "DMR", "LMG",
    "Spring", "CO2", "HPA", "Accessories", "Optics", "Gear", "Clothing",
    "Batteries", "BBs", "Magazines", "Upgrade Parts", "Hop-up", "Barrel",
    "Motor", "Gearbox", "Piston", "Cylinder", "Nozzle", "Spring Guide",
    "Rail", "Handguard", "Stock", "Grip", "Muzzle", "Suppressor", "Tracer",
    "Red Dot", "Scope", "Holster", "Vest", "Plate Carrier", "Helmet",
    "Gloves", "Boots", "Goggles", "Mask", "Radio", "Grenades",
]


@dataclass
class SourceConfig:
    url: str
    name: str
    slug: str
    type: str  # reddit, forum, blog, guide, brand_site, news
    max_pages: int = 100
    max_depth: int = 2
    crawl_delay: float = 1.5
    focus_topics: List[str] = field(default_factory=list)


@dataclass
class ExtractedTopic:
    title: str
    summary: str
    brands: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    source_url: str = ""
    sentiment: str = "neutral"  # positive, negative, neutral, mixed
    topic_type: str = "discussion"  # discussion, review, guide, news, release


@dataclass
class SourceResult:
    config: SourceConfig
    topics: List[ExtractedTopic] = field(default_factory=list)
    brands_mentioned: List[str] = field(default_factory=list)
    categories_mentioned: List[str] = field(default_factory=list)
    pages_crawled: int = 0
    summary: str = ""
    error: Optional[str] = None


@dataclass
class ContentLearnResult:
    started_at: str = ""
    completed_at: str = ""
    sources_processed: int = 0
    sources_failed: int = 0
    notes_written: int = 0
    guides_extracted: int = 0
    trends_detected: int = 0
    errors: List[str] = field(default_factory=list)
    source_results: Dict[str, dict] = field(default_factory=dict)


def _slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")[:80]


# ── Human-edit preservation ──────────────────────────────────

_HUMAN_BLOCK_RE = re.compile(
    r"(<!-- human-start -->.*?<!-- human-end -->)",
    re.DOTALL,
)


def _extract_human_blocks(content: str) -> List[str]:
    return _HUMAN_BLOCK_RE.findall(content)


def _inject_human_blocks(new_content: str, old_blocks: List[str]) -> str:
    new_blocks = _HUMAN_BLOCK_RE.findall(new_content)
    for i, old_block in enumerate(old_blocks):
        if i < len(new_blocks):
            new_content = new_content.replace(new_blocks[i], old_block, 1)
    return new_content


class ContentLearner:
    """Crawls content sources, summarizes with LLM, generates vault notes."""

    def __init__(
        self,
        vault_path: str,
        config_path: str = None,
        llm_client=None,
        redis_client=None,
        competitor_indexer=None,  # reuse WebIndexer for crawling
    ):
        self.vault_path = Path(vault_path)
        self.llm_client = llm_client
        self.redis = redis_client
        self._crawler = competitor_indexer  # WebIndexer instance

        # Load sources from config
        self.sources: List[SourceConfig] = []
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            for s in cfg.get("sources", []):
                self.sources.append(SourceConfig(**s))
            logger.info(f"ContentLearner: loaded {len(self.sources)} sources")

        # Jinja2 environment
        templates_dir = self.vault_path / "_templates"
        if templates_dir.exists():
            self.jinja = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            self.jinja.filters["format_wikilink"] = lambda s: f"[[{s}]]"
        else:
            self.jinja = None

    async def learn(self, sources: List[SourceConfig] = None) -> ContentLearnResult:
        """Process all content sources: crawl → summarize → extract → write notes."""
        result = ContentLearnResult(
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        targets = sources or self.sources
        if not targets:
            result.errors.append("No content sources configured")
            result.completed_at = datetime.now(timezone.utc).isoformat()
            return result

        # Process each source
        all_results: Dict[str, SourceResult] = {}
        for config in targets:
            logger.info(f"Learning from: {config.name} ({config.url})")
            try:
                sr = await self._process_source(config)
                all_results[config.slug] = sr
                result.sources_processed += 1
                result.source_results[config.slug] = {
                    "name": config.name,
                    "type": config.type,
                    "topics": len(sr.topics),
                    "brands": len(sr.brands_mentioned),
                    "pages": sr.pages_crawled,
                    "error": sr.error,
                }
            except Exception as e:
                logger.error(f"Failed to learn from {config.name}: {e}")
                result.sources_failed += 1
                result.errors.append(f"{config.name}: {str(e)[:200]}")

        if not all_results:
            result.errors.append("No sources processed successfully")
            result.completed_at = datetime.now(timezone.utc).isoformat()
            return result

        # Generate notes per source
        for slug, sr in all_results.items():
            result.notes_written += self._write_source_note(sr)

        # Extract guides from guide-type sources
        guides = self._extract_guides(all_results)
        for guide in guides:
            result.notes_written += self._write_guide_note(guide)
        result.guides_extracted = len(guides)

        # Detect trends across all sources
        trends = self._detect_trends(all_results)
        for trend in trends:
            result.notes_written += self._write_trend_note(trend)
        result.trends_detected = len(trends)

        # Community digest from forum/reddit sources
        community_sources = {s: r for s, r in all_results.items()
                            if r.config.type in ("reddit", "forum")}
        if community_sources:
            result.notes_written += self._write_community_digest(community_sources)

        # Update brand notes with source mentions
        self._enrich_brand_mentions(all_results)

        result.completed_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Content learning complete: {result.sources_processed} sources, "
            f"{result.notes_written} notes, {result.guides_extracted} guides, "
            f"{result.trends_detected} trends"
        )
        return result

    async def _process_source(self, config: SourceConfig) -> SourceResult:
        """Crawl a content source and extract knowledge from it."""
        sr = SourceResult(config=config)

        # ── 1. Crawl the source ──
        if self._crawler:
            try:
                job = await self._crawler.crawl_and_index(
                    start_url=config.url,
                    max_depth=config.max_depth,
                    max_pages=config.max_pages,
                )
                sr.pages_crawled = job.pages_indexed
                logger.info(f"Crawled {config.name}: {job.pages_indexed} pages")
            except Exception as e:
                logger.warning(f"Crawl failed for {config.name}: {e}")
                sr.error = f"Crawl failed: {str(e)[:200]}"
                # Continue — try to use any previously crawled content
        else:
            sr.error = "No crawler available"
            return sr

        # ── 2. Retrieve crawled chunks from Qdrant ──
        chunks = await self._get_crawled_chunks(config)
        if not chunks:
            logger.warning(f"No chunks found for {config.name}")
            sr.error = "No content retrieved after crawl"
            return sr

        # ── 3. LLM: Summarize and extract entities from chunks ──
        if self.llm_client:
            # Process in batches to manage LLM rate limits
            batch_size = 5
            all_topics = []
            all_brands = set()
            all_categories = set()

            for i in range(0, min(len(chunks), 50), batch_size):  # Process up to 50 chunks
                batch = chunks[i:i + batch_size]
                batch_text = "\n\n---\n\n".join(
                    f"[Page: {c.get('url', 'unknown')}]\n{c.get('content', '')[:1500]}"
                    for c in batch
                )

                try:
                    extracted = await self._llm_extract_knowledge(
                        batch_text, config.name, config.type, config.focus_topics
                    )
                    if extracted:
                        for topic in extracted.get("topics", []):
                            et = ExtractedTopic(
                                title=topic.get("title", ""),
                                summary=topic.get("summary", ""),
                                brands=[_slugify(b) for b in topic.get("brands", [])],
                                categories=[_slugify(c) for c in topic.get("categories", [])],
                                source_url=topic.get("source_url", ""),
                                sentiment=topic.get("sentiment", "neutral"),
                                topic_type=topic.get("type", "discussion"),
                            )
                            all_topics.append(et)
                        all_brands.update(_slugify(b) for b in extracted.get("brands", []))
                        all_categories.update(_slugify(c) for c in extracted.get("categories", []))
                except Exception as e:
                    logger.warning(f"LLM extraction failed for batch {i}: {e}")

            sr.topics = all_topics
            sr.brands_mentioned = sorted(all_brands)
            sr.categories_mentioned = sorted(all_categories)

            # Generate source summary
            if sr.topics:
                try:
                    sr.summary = await self._llm_summarize_source(config, sr.topics)
                except Exception as e:
                    logger.warning(f"Summary generation failed: {e}")
                    sr.summary = f"Content source with {len(sr.topics)} topics extracted."
        else:
            # No LLM — do basic entity extraction via regex
            sr.brands_mentioned = self._regex_extract_brands(chunks)
            sr.categories_mentioned = self._regex_extract_categories(chunks)
            sr.summary = f"Content source with {len(chunks)} pages crawled."

        return sr

    async def _get_crawled_chunks(self, config: SourceConfig) -> List[dict]:
        """Retrieve crawled content chunks from Qdrant for this source."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue

            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            client = QdrantClient(url=qdrant_url)

            domain = config.url.replace("https://", "").replace("http://", "").split("/")[0]
            # Try with and without www
            domains = [domain]
            if not domain.startswith("www."):
                domains.append(f"www.{domain}")

            all_chunks = []
            for d in domains:
                try:
                    # Search in web_pages collection (general content)
                    results = client.scroll(
                        collection_name="web_pages",
                        scroll_filter=Filter(
                            must=[FieldCondition(key="domain", match=MatchValue(value=d))]
                        ),
                        limit=100,
                        with_payload=True,
                    )
                    if results and results[0]:
                        for point in results[0]:
                            all_chunks.append(point.payload)
                except Exception:
                    pass

                try:
                    # Also check competitor_products collection
                    results = client.scroll(
                        collection_name="competitor_products_v2",
                        scroll_filter=Filter(
                            must=[FieldCondition(key="domain", match=MatchValue(value=d))]
                        ),
                        limit=100,
                        with_payload=True,
                    )
                    if results and results[0]:
                        for point in results[0]:
                            all_chunks.append(point.payload)
                except Exception:
                    pass

            return all_chunks

        except Exception as e:
            logger.warning(f"Failed to retrieve chunks for {config.name}: {e}")
            return []

    async def _llm_extract_knowledge(
        self,
        text: str,
        source_name: str,
        source_type: str,
        focus_topics: List[str],
    ) -> Optional[dict]:
        """Use LLM to extract structured knowledge from crawled content."""
        if not self.llm_client:
            return None

        focus = ", ".join(focus_topics) if focus_topics else "airsoft products, brands, guides, community"

        prompt = f"""Analyze this airsoft content from {source_name} ({source_type}).

Extract structured knowledge as JSON:
{{
  "topics": [
    {{
      "title": "short topic title",
      "summary": "2-3 sentence summary of the key information",
      "brands": ["brand names mentioned"],
      "categories": ["product categories discussed"],
      "sentiment": "positive/negative/neutral/mixed",
      "type": "discussion/review/guide/news/release",
      "source_url": "url if identifiable"
    }}
  ],
  "brands": ["all brand names found"],
  "categories": ["all product categories found"]
}}

Focus on: {focus}
Only include topics that contain actionable airsoft market intelligence.
Return valid JSON only, no markdown.

Content:
{text[:4000]}"""

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "local"),
                    messages=[
                        {"role": "system", "content": "You are an airsoft market intelligence analyst. Extract structured knowledge from content. Return ONLY valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2000,
                    timeout=120,
                ),
            )
            text_resp = response.choices[0].message.content.strip()
            # Strip thinking tokens
            text_resp = re.sub(r"<think>.*?</think>", "", text_resp, flags=re.DOTALL).strip()
            text_resp = re.sub(r"^Thinking Process:.*?(?=\{)", "", text_resp, flags=re.DOTALL).strip()
            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", text_resp)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            logger.warning(f"LLM knowledge extraction failed: {e}")
            return None

    async def _llm_summarize_source(self, config: SourceConfig, topics: List[ExtractedTopic]) -> str:
        """Generate an overall summary for a content source."""
        topic_list = "\n".join(f"- {t.title}: {t.summary[:100]}" for t in topics[:15])
        prompt = (
            f"Write a 3-4 sentence overview of what we learned from {config.name} ({config.type}). "
            f"Topics found:\n{topic_list}\n\n"
            f"Focus on what's most relevant for an airsoft retailer in Spain."
        )

        if not self.llm_client:
            return f"Content source with {len(topics)} topics."

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "local"),
                    messages=[
                        {"role": "system", "content": "You are a concise airsoft market analyst. 3-4 sentences max."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=300,
                    timeout=60,
                ),
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            text = re.sub(r"^Thinking Process:.*?(?=\n[A-Z]|\n\n[^\s])", "", text, flags=re.DOTALL).strip()
            return text or f"Content source with {len(topics)} topics."
        except Exception as e:
            return f"Content source with {len(topics)} topics."

    def _regex_extract_brands(self, chunks: List[dict]) -> List[str]:
        """Fallback brand extraction using regex matching."""
        found = set()
        all_text = " ".join(c.get("content", "") for c in chunks[:50]).lower()
        for brand in KNOWN_BRANDS:
            if brand.lower() in all_text:
                found.add(_slugify(brand))
        return sorted(found)

    def _regex_extract_categories(self, chunks: List[dict]) -> List[str]:
        """Fallback category extraction using regex matching."""
        found = set()
        all_text = " ".join(c.get("content", "") for c in chunks[:50]).lower()
        for cat in KNOWN_CATEGORIES:
            if cat.lower() in all_text:
                found.add(_slugify(cat))
        return sorted(found)

    # ── Note writing ──────────────────────────────────────────

    def _write_note(self, path: Path, content: str) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old_content = path.read_text(encoding="utf-8")
            old_blocks = _extract_human_blocks(old_content)
            if old_blocks:
                content = _inject_human_blocks(content, old_blocks)
        path.write_text(content, encoding="utf-8")
        return True

    def _write_source_note(self, sr: SourceResult) -> int:
        """Write a source note from template."""
        if not self.jinja or sr.error:
            return 0

        # Map source type to subdirectory
        type_dirs = {
            "reddit": "sources/reddit",
            "forum": "sources/forums",
            "blog": "sources/blogs",
            "guide": "guides",
            "brand_site": "brands",
            "news": "sources/blogs",
        }
        subdir = type_dirs.get(sr.config.type, "sources")

        template = self.jinja.get_template("source.md.j2")
        content = template.render(
            name=sr.config.name,
            slug=sr.config.slug,
            source_type=sr.config.type,
            url=sr.config.url,
            last_crawled=datetime.now(timezone.utc).isoformat(),
            pages_indexed=sr.pages_crawled,
            topics=sr.topics,
            summary=sr.summary or f"Content source with {sr.pages_crawled} pages.",
            brands_mentioned=sr.brands_mentioned,
            categories_mentioned=sr.categories_mentioned,
            related_sources=[],
        )

        path = self.vault_path / subdir / f"{sr.config.slug}.md"
        self._write_note(path, content)
        logger.info(f"Wrote source note: {path}")
        return 1

    def _extract_guides(self, all_results: Dict[str, SourceResult]) -> List[dict]:
        """Extract guide-worthy content from processed sources."""
        guides = []
        for slug, sr in all_results.items():
            for topic in sr.topics:
                if topic.topic_type in ("guide", "review") and len(topic.summary) > 50:
                    guides.append({
                        "title": topic.title,
                        "slug": _slugify(topic.title),
                        "source_name": sr.config.name,
                        "source_url": topic.source_url or sr.config.url,
                        "summary": topic.summary,
                        "key_points": [],
                        "brands": topic.brands,
                        "categories": topic.categories,
                        "recommendations": [],
                        "difficulty": "intermediate",
                    })
        return guides[:20]  # Cap at 20 guides per run

    def _write_guide_note(self, guide: dict) -> int:
        if not self.jinja:
            return 0

        template = self.jinja.get_template("guide.md.j2")
        content = template.render(
            last_updated=datetime.now(timezone.utc).isoformat(),
            **guide,
        )

        path = self.vault_path / "guides" / f"{guide['slug']}.md"
        self._write_note(path, content)
        return 1

    def _detect_trends(self, all_results: Dict[str, SourceResult]) -> List[dict]:
        """Detect trends from aggregated source data."""
        # Brand mention frequency across sources
        brand_freq = defaultdict(lambda: {"count": 0, "sources": set(), "sentiment": []})
        for slug, sr in all_results.items():
            for topic in sr.topics:
                for brand in topic.brands:
                    brand_freq[brand]["count"] += 1
                    brand_freq[brand]["sources"].add(slug)
                    brand_freq[brand]["sentiment"].append(topic.sentiment)

        trends = []
        # Brands trending across multiple sources
        for brand, data in brand_freq.items():
            if len(data["sources"]) >= 2 and data["count"] >= 3:
                sentiments = data["sentiment"]
                pos = sentiments.count("positive")
                neg = sentiments.count("negative")
                overall = "positive" if pos > neg else "negative" if neg > pos else "mixed"

                trends.append({
                    "title": f"{brand} — Trending Across Sources",
                    "slug": f"trend-{brand}",
                    "detected_date": datetime.now(timezone.utc).isoformat(),
                    "trend_type": "brand_momentum",
                    "confidence": "high" if data["count"] >= 5 else "medium",
                    "description": f"{brand} mentioned {data['count']} times across {len(data['sources'])} sources with {overall} sentiment.",
                    "brands": [brand],
                    "categories": [],
                    "sources": sorted(data["sources"]),
                    "evidence_points": [
                        {"source": src, "detail": f"Mentioned in content"} for src in data["sources"]
                    ],
                    "action_items": [
                        f"Review {brand} product coverage in our catalog",
                        f"Check pricing competitiveness for {brand} products",
                    ],
                })

        return trends[:10]  # Top 10 trends

    def _write_trend_note(self, trend: dict) -> int:
        if not self.jinja:
            return 0

        template = self.jinja.get_template("trend.md.j2")
        content = template.render(**trend)

        path = self.vault_path / "trends" / f"{trend['slug']}.md"
        self._write_note(path, content)
        return 1

    def _write_community_digest(self, community_sources: Dict[str, SourceResult]) -> int:
        """Write a community digest from Reddit/forum sources."""
        if not self.jinja:
            return 0

        all_topics = []
        all_brands = set()
        for slug, sr in community_sources.items():
            for t in sr.topics:
                all_topics.append({
                    "title": t.title,
                    "summary": t.summary,
                    "sentiment": t.sentiment,
                    "url": t.source_url,
                })
            all_brands.update(sr.brands_mentioned)

        # Find frequently recommended items
        recommended = []
        for topic in all_topics:
            if topic.get("sentiment") == "positive":
                recommended.append({
                    "name": topic["title"][:60],
                    "reason": topic["summary"][:100],
                    "brand": "",
                })

        template = self.jinja.get_template("community.md.j2")
        content = template.render(
            title="Community Pulse — Airsoft Discussions",
            slug="community-pulse",
            platform="Reddit + Forums",
            last_updated=datetime.now(timezone.utc).isoformat(),
            discussion_count=len(all_topics),
            sentiment="mixed",
            summary=f"Aggregated insights from {len(community_sources)} community sources with {len(all_topics)} discussions.",
            hot_topics=all_topics[:15],
            frequently_recommended=recommended[:10],
            complaints=[t["summary"][:100] for t in all_topics if t.get("sentiment") == "negative"][:10],
            brands=sorted(all_brands),
        )

        path = self.vault_path / "community" / "community-pulse.md"
        self._write_note(path, content)
        return 1

    def _enrich_brand_mentions(self, all_results: Dict[str, SourceResult]):
        """Update existing brand notes with source mention data."""
        brand_dir = self.vault_path / "brands"
        if not brand_dir.exists():
            return

        # Collect brand mentions across sources
        mentions = defaultdict(list)
        for slug, sr in all_results.items():
            for brand in sr.brands_mentioned:
                mentions[brand].append({
                    "source": sr.config.name,
                    "type": sr.config.type,
                    "topics": [t for t in sr.topics if brand in t.brands],
                })

        # For each brand with existing note, append source mentions to frontmatter
        for brand_file in brand_dir.glob("*.md"):
            brand_slug = brand_file.stem
            if brand_slug in mentions:
                logger.info(f"Brand {brand_slug} mentioned in {len(mentions[brand_slug])} content sources")
                # Don't modify the file — the vault_builder handles brand notes
                # Just log for now; future: update frontmatter with source refs

    def get_status(self) -> dict:
        """Get content learner status."""
        note_count = 0
        for d in ("sources", "guides", "trends", "community"):
            dir_path = self.vault_path / d
            if dir_path.exists():
                note_count += len(list(dir_path.rglob("*.md")))

        return {
            "sources_configured": len(self.sources),
            "content_notes": note_count,
            "source_types": list(set(s.type for s in self.sources)),
        }
