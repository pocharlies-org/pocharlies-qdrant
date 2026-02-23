"""
Translation Pipeline for RAG Pipeline
LLM-based translation with multi-language airsoft terminology glossary.
Supports 20 languages with English as hub language.
"""

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Callable

from glossary_data import GLOSSARY, SUPPORTED_LANGUAGES, get_glossary_for_pair

logger = logging.getLogger(__name__)


@dataclass
class TranslationJob:
    """Tracks a translation job."""
    job_id: str
    source_lang: str = "en"
    target_lang: str = "es"
    status: str = "running"
    items_processed: int = 0
    items_total: int = 0
    logs: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    MAX_LOGS = 2000

    def log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")
        if len(self.logs) > self.MAX_LOGS:
            self.logs = self.logs[:50] + self.logs[-(self.MAX_LOGS - 50):]

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "status": self.status,
            "items_processed": self.items_processed,
            "items_total": self.items_total,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


TRANSLATION_SYSTEM_PROMPT = """You are a professional translator specializing in airsoft and tactical equipment.
Translate the following product descriptions from {source} to {target}.

Rules:
- Preserve brand names (Tokyo Marui, WE-Tech, G&G, KWA, etc.) untranslated
- Preserve model numbers (M4A1, Hi-Capa 5.1, MP5, AK-47, etc.) untranslated
- Preserve technical measurements (6mm, 350 FPS, 1.2J, etc.) untranslated
- Preserve SKU/part numbers untranslated
- Maintain the same formatting and paragraph structure
- Use natural, fluent {target_name} appropriate for an e-commerce product listing
- Return ONLY the translations as a JSON array of strings, matching input order
- The array must have exactly the same number of elements as the input"""


# ── Spec Normalization ──────────────────────────────────────────

# Unit conversion functions
UNIT_CONVERSIONS = {
    "fps_to_ms": lambda v: round(v * 0.3048, 1),
    "ms_to_fps": lambda v: round(v / 0.3048),
    "oz_to_g": lambda v: round(v * 28.3495, 1),
    "g_to_oz": lambda v: round(v / 28.3495, 2),
    "lbs_to_kg": lambda v: round(v * 0.453592, 2),
    "kg_to_lbs": lambda v: round(v / 0.453592, 2),
    "inch_to_mm": lambda v: round(v * 25.4, 1),
    "mm_to_inch": lambda v: round(v / 25.4, 2),
}

# Brand name normalization
BRAND_NORMALIZATIONS = {
    "tokyo marui": "Tokyo Marui",
    "tm": "Tokyo Marui",
    "we tech": "WE-Tech",
    "we-tech": "WE-Tech",
    "wetech": "WE-Tech",
    "g&g": "G&G Armament",
    "g & g": "G&G Armament",
    "gg armament": "G&G Armament",
    "kwa": "KWA",
    "ksc": "KSC",
    "vfc": "VFC",
    "elite force": "Elite Force",
    "eliteforce": "Elite Force",
    "ares": "ARES",
    "ics": "ICS",
    "krytac": "Krytac",
    "lct": "LCT",
    "e&l": "E&L",
    "cyma": "CYMA",
    "jg": "JG",
    "jing gong": "JG",
    "classic army": "Classic Army",
    "asg": "ASG",
    "action sport games": "ASG",
    "nuprol": "Nuprol",
    "novritsch": "Novritsch",
    "modify": "Modify",
    "maxx": "MAXX",
    "gate": "GATE",
    "maple leaf": "Maple Leaf",
    "prometheus": "Prometheus",
    "pdi": "PDI",
    "guarder": "Guarder",
    "shs": "SHS",
    "lonex": "Lonex",
    "retro arms": "Retro Arms",
}

# Re-export for backward compat; canonical source is glossary_data.py
LANG_NAMES = SUPPORTED_LANGUAGES


class GlossaryStore:
    """Redis-backed store for user-defined glossary terms.

    Custom entries are stored per language pair:
      Redis key: translation:glossary:{source}:{target}
    Legacy entries (from old single-key store) are auto-migrated on first load.
    """

    KEY_PREFIX = "translation:glossary"
    LEGACY_KEY = "translation:glossary"  # old single-hash key (en:es assumed)

    def __init__(self, redis=None):
        self.redis = redis
        # Per-pair cache: {(source, target): {term: translation}}
        self._cache: Dict[tuple, Dict[str, str]] = {}

    def _redis_key(self, source_lang: str, target_lang: str) -> str:
        return f"{self.KEY_PREFIX}:{source_lang}:{target_lang}"

    async def _migrate_legacy(self):
        """Migrate old single-key glossary to per-pair format (en:es)."""
        if not self.redis:
            return
        try:
            raw = await self.redis.hgetall(self.LEGACY_KEY)
            if not raw:
                return
            # Check if it's the old format (no colon-separated lang pairs in key name)
            new_key = self._redis_key("en", "es")
            exists = await self.redis.exists(new_key)
            if not exists and raw:
                decoded = {
                    (k.decode() if isinstance(k, bytes) else k):
                    (v.decode() if isinstance(v, bytes) else v)
                    for k, v in raw.items()
                }
                if decoded:
                    await self.redis.hset(new_key, mapping=decoded)
                    logger.info(f"Migrated {len(decoded)} legacy glossary entries to {new_key}")
        except Exception as e:
            logger.warning(f"Legacy glossary migration failed: {e}")

    async def load(self, source_lang: str = "en", target_lang: str = "es") -> Dict[str, str]:
        """Load custom glossary entries for a language pair from Redis."""
        pair = (source_lang, target_lang)
        if not self.redis:
            return self._cache.get(pair, {})
        try:
            key = self._redis_key(source_lang, target_lang)
            raw = await self.redis.hgetall(key)
            entries = {
                (k.decode() if isinstance(k, bytes) else k):
                (v.decode() if isinstance(v, bytes) else v)
                for k, v in raw.items()
            }
            self._cache[pair] = entries
        except Exception as e:
            logger.warning(f"Failed to load glossary ({source_lang}→{target_lang}) from Redis: {e}")
        return self._cache.get(pair, {})

    async def load_all_pairs(self) -> int:
        """Load all custom glossary pairs from Redis. Returns total custom entry count."""
        if not self.redis:
            return 0
        await self._migrate_legacy()
        total = 0
        try:
            pattern = f"{self.KEY_PREFIX}:*:*"
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                k = key.decode() if isinstance(key, bytes) else key
                # Parse "translation:glossary:en:es" → ("en", "es")
                parts = k.split(":")
                if len(parts) == 4:
                    keys.append((parts[2], parts[3]))
            for src, tgt in keys:
                entries = await self.load(src, tgt)
                total += len(entries)
        except Exception as e:
            logger.warning(f"Failed to scan glossary keys: {e}")
        return total

    async def add(self, source_term: str, target_term: str,
                  source_lang: str = "en", target_lang: str = "es") -> None:
        """Add or update a custom glossary entry for a language pair."""
        pair = (source_lang, target_lang)
        key = source_term.strip().lower()
        val = target_term.strip()
        if pair not in self._cache:
            self._cache[pair] = {}
        self._cache[pair][key] = val
        if self.redis:
            try:
                await self.redis.hset(self._redis_key(source_lang, target_lang), key, val)
            except Exception as e:
                logger.warning(f"Failed to save glossary entry to Redis: {e}")

    async def add_bulk(self, entries: Dict[str, str],
                       source_lang: str = "en", target_lang: str = "es") -> int:
        """Add multiple glossary entries for a language pair. Returns count added."""
        pair = (source_lang, target_lang)
        clean = {k.strip().lower(): v.strip() for k, v in entries.items() if k.strip() and v.strip()}
        if pair not in self._cache:
            self._cache[pair] = {}
        self._cache[pair].update(clean)
        if self.redis and clean:
            try:
                await self.redis.hset(self._redis_key(source_lang, target_lang), mapping=clean)
            except Exception as e:
                logger.warning(f"Failed to bulk save glossary to Redis: {e}")
        return len(clean)

    async def remove(self, source_term: str,
                     source_lang: str = "en", target_lang: str = "es") -> bool:
        """Remove a custom glossary entry. Returns True if existed."""
        pair = (source_lang, target_lang)
        key = source_term.strip().lower()
        existed = key in self._cache.get(pair, {})
        if pair in self._cache:
            self._cache[pair].pop(key, None)
        if self.redis:
            try:
                await self.redis.hdel(self._redis_key(source_lang, target_lang), key)
            except Exception as e:
                logger.warning(f"Failed to remove glossary entry from Redis: {e}")
        return existed

    async def get_all(self, source_lang: str = "en", target_lang: str = "es") -> Dict[str, str]:
        """Get all custom entries for a language pair (cached)."""
        pair = (source_lang, target_lang)
        if pair not in self._cache and self.redis:
            await self.load(source_lang, target_lang)
        return dict(self._cache.get(pair, {}))

    def get_relevant(self, text: str, source_lang: str, target_lang: str) -> Dict[str, str]:
        """Find glossary entries relevant to the input text.

        Checks both built-in (from glossary_data) and custom glossary,
        returns only terms that appear in the text.
        Works for ANY supported language pair.
        """
        text_lower = text.lower()
        relevant = {}

        # Get built-in terms for this language pair (works for all 20 languages)
        builtin = get_glossary_for_pair(source_lang, target_lang)

        # Check built-in terms (longest first to avoid partial matches)
        for term in sorted(builtin.keys(), key=len, reverse=True):
            if term in text_lower:
                relevant[term] = builtin[term]

        # Check custom terms for this pair (override built-in if same key)
        pair = (source_lang, target_lang)
        for term, translation in self._cache.get(pair, {}).items():
            if term in text_lower:
                relevant[term] = translation

        return relevant


def estimate_tokens(text: str) -> int:
    """Estimate token count. Conservative 1.4x multiplier for subword tokenization."""
    return max(1, int(len(text.split()) * 1.4))


def pack_batches(texts: List[str], max_input_tokens: int = 12000, min_batch: int = 3) -> List[List[int]]:
    """Greedy first-fit bin packing by estimated token count.

    Returns list of batches, each batch is a list of text indices.
    Each batch targets max_input_tokens. A single text exceeding the
    budget gets its own batch (never split a text across calls).
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for i, text in enumerate(texts):
        tokens = estimate_tokens(text)
        if current_batch and current_tokens + tokens > max_input_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(i)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)
    return batches


class TranslationPipeline:
    def __init__(self, llm_client, glossary_store: Optional[GlossaryStore] = None):
        self.llm_client = llm_client
        self.glossary = glossary_store or GlossaryStore()

    MAX_CONCURRENT_CHUNKS = 32

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "en",
        target_lang: str = "es",
        progress_callback: Optional[Callable] = None,
        rag_context: Optional[str] = None,
    ) -> TranslationJob:
        """Translate a list of texts using the running LLM.

        Sends up to 8 chunks to the LLM concurrently for throughput.
        """
        job = TranslationJob(
            job_id=uuid.uuid4().hex[:12],
            source_lang=source_lang,
            target_lang=target_lang,
            items_total=len(texts),
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            chunk_size = 5
            chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
            semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_CHUNKS)

            async def _translate_one(chunk_idx: int, chunk: List[str]) -> tuple:
                async with semaphore:
                    result = await self._translate_chunk(
                        chunk, source_lang, target_lang, rag_context=rag_context
                    )
                    return chunk_idx, result

            tasks = [_translate_one(idx, chunk) for idx, chunk in enumerate(chunks)]
            results_ordered = [None] * len(chunks)

            for coro in asyncio.as_completed(tasks):
                idx, translated = await coro
                results_ordered[idx] = translated
                job.items_processed = sum(
                    len(r) for r in results_ordered if r is not None
                )
                job.log(f"Translated {job.items_processed}/{job.items_total}")
                if progress_callback:
                    progress_callback(job)

            # Flatten in order
            for result in results_ordered:
                job.results.extend(result or [])

            job.status = "completed"
            job.log(f"COMPLETED: {len(job.results)} items translated")

        except Exception as e:
            job.status = "failed"
            job.log(f"FAILED: {str(e)[:200]}")
            logger.error(f"Translation failed: {e}")

        job.ended_at = datetime.now(timezone.utc).isoformat()
        if progress_callback:
            progress_callback(job)
        return job

    def _build_glossary_prompt(self, texts: List[str], source_lang: str, target_lang: str) -> str:
        """Build a glossary section for the prompt based on terms found in the input."""
        combined_text = " ".join(texts)
        relevant = self.glossary.get_relevant(combined_text, source_lang, target_lang)

        if not relevant:
            return ""

        lines = [f"  - \"{term}\" → \"{translation}\"" for term, translation in relevant.items()]
        return (
            "\n\n## MANDATORY Terminology Glossary\n"
            "You MUST use these exact translations for the following terms. "
            "Do NOT deviate from this glossary:\n"
            + "\n".join(lines)
        )

    async def _translate_chunk(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        rag_context: Optional[str] = None,
    ) -> List[str]:
        """Single LLM call to translate a small batch of texts."""
        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)

        system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
            source=source_name,
            target=target_name,
            target_name=target_name,
        )

        # Inject relevant glossary terms
        glossary_section = self._build_glossary_prompt(texts, source_lang, target_lang)
        if glossary_section:
            system_prompt += glossary_section

        if rag_context:
            system_prompt += f"\n\n## Reference: Similar products from our catalog\n{rag_context}"

        # Number the texts for clarity
        numbered = "\n\n".join(
            f"[{i+1}] {text}" for i, text in enumerate(texts)
        )

        try:
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(None, self.llm_client.models.list)
            model_id = models.data[0].id if models.data else None

            if not model_id:
                return texts  # fallback: return originals

            response = await loop.run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate these {len(texts)} texts:\n\n{numbered}"},
                    ],
                    max_tokens=4096,
                    temperature=0.2,
                    timeout=300,
                    user="rag:translate",
                ),
            )

            response_text = response.choices[0].message.content
            return self._parse_translations(response_text, len(texts), texts)

        except Exception as e:
            logger.warning(f"Translation chunk failed: {e}")
            return texts  # fallback

    @staticmethod
    def _parse_translations(response_text: str, expected_count: int, originals: List[str]) -> List[str]:
        """Parse LLM translation response into list of strings."""
        text = response_text.strip()

        # Try JSON array parse
        if "```" in text:
            lines = text.split("\n")
            inside = False
            json_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    inside = not inside
                    continue
                if inside:
                    json_lines.append(line)
            if json_lines:
                text = "\n".join(json_lines)

        try:
            import json
            data = json.loads(text)
            if isinstance(data, list) and len(data) == expected_count:
                return [str(item) for item in data]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to split by numbered markers [1], [2], etc.
        parts = re.split(r"\[\d+\]\s*", text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) == expected_count:
            return parts

        # Last resort: return originals
        logger.warning(f"Could not parse {expected_count} translations from LLM response")
        return originals

    # ── Spec Normalization ────────────────────────────────────────

    @staticmethod
    def normalize_specs(product: dict, target_units: str = "metric") -> dict:
        """Normalize units, brand names, and model numbers in a product spec dict."""
        result = dict(product)

        # Normalize brand
        brand = result.get("brand", "")
        if brand:
            result["brand"] = TranslationPipeline.normalize_brand(brand)

        # Normalize FPS/m/s
        fps = result.get("fps")
        if fps and target_units == "metric":
            result["velocity_ms"] = UNIT_CONVERSIONS["fps_to_ms"](fps)

        # Normalize weight
        weight = result.get("weight_grams")
        if weight and target_units == "metric":
            if weight > 0:
                result["weight_kg"] = round(weight / 1000, 2)

        return result

    @staticmethod
    def normalize_brand(brand: str) -> str:
        """Fuzzy match and normalize brand name."""
        brand_lower = brand.lower().strip()
        return BRAND_NORMALIZATIONS.get(brand_lower, brand)
