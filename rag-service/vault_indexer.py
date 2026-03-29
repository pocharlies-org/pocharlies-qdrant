"""
Vault Indexer — Reads Obsidian vault markdown files and indexes them into Qdrant.

Uses BGE dense + BM25 sparse hybrid embedding (same as catalog_indexer).
Content-hash dedup via Redis to avoid re-embedding unchanged notes.
"""

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import frontmatter

from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
    SparseVectorParams, SparseIndexParams,
    PayloadSchemaType,
    Prefetch, FusionQuery, Fusion,
)

from qdrant_utils import make_qdrant_client
import bgem3_encoder

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_brain"
HASH_PREFIX = "vault:hash"

# Directories to exclude from indexing
EXCLUDE_DIRS = {"_templates", "_meta", ".obsidian"}

# Wiki-link pattern
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

# Markdown heading pattern
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def _generate_point_id(note_path: str, chunk_index: int) -> int:
    """Generate a deterministic Qdrant point ID from note path + chunk index."""
    key = f"vault:{note_path}:{chunk_index}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:16], 16)


def _chunk_markdown(content: str, max_chunk_size: int = 1000) -> List[Dict]:
    """Split markdown by heading boundaries.

    Returns list of dicts with: section_heading, text, chunk_index.
    Small sections are merged with the next to avoid tiny chunks.
    """
    # Split content into sections by headings
    parts = HEADING_RE.split(content)

    sections = []
    current_heading = "Introduction"
    current_text_parts = []

    i = 0
    while i < len(parts):
        part = parts[i]

        # Check if this is a heading marker (# or ## etc.)
        if i + 1 < len(parts) and re.match(r"^#{1,4}$", part.strip()):
            # Save previous section
            if current_text_parts:
                text = "\n".join(current_text_parts).strip()
                if text and len(text) > 20:
                    sections.append({"section_heading": current_heading, "text": text})

            # New section
            current_heading = parts[i + 1].strip()
            current_text_parts = []
            i += 2
            continue

        # Regular text
        text = part.strip()
        if text:
            current_text_parts.append(text)
        i += 1

    # Save last section
    if current_text_parts:
        text = "\n".join(current_text_parts).strip()
        if text and len(text) > 20:
            sections.append({"section_heading": current_heading, "text": text})

    # Split oversized sections
    final_sections = []
    for section in sections:
        if len(section["text"]) > max_chunk_size:
            paragraphs = section["text"].split("\n\n")
            chunk_text = ""
            for para in paragraphs:
                if len(chunk_text) + len(para) > max_chunk_size and chunk_text:
                    final_sections.append({
                        "section_heading": section["section_heading"],
                        "text": chunk_text.strip(),
                    })
                    chunk_text = para
                else:
                    chunk_text = f"{chunk_text}\n\n{para}" if chunk_text else para
            if chunk_text.strip():
                final_sections.append({
                    "section_heading": section["section_heading"],
                    "text": chunk_text.strip(),
                })
        else:
            final_sections.append(section)

    # Add chunk indices
    for idx, section in enumerate(final_sections):
        section["chunk_index"] = idx

    return final_sections


class VaultIndexer:
    """Indexes Obsidian vault markdown files into a Qdrant collection."""

    def __init__(
        self,
        vault_path: str,
        qdrant_url: str,
        qdrant_api_key: str = None,
        model=None,
        embedding_model: str = "BAAI/bge-m3",
        redis_client=None,
    ):
        self.vault_path = Path(vault_path)
        self.client = make_qdrant_client(qdrant_url, qdrant_api_key)
        self.redis = redis_client
        self.dim = bgem3_encoder.DENSE_DIM

    def _ensure_collection(self):
        """Create knowledge_brain collection if it doesn't exist."""
        existing = [c.name for c in self.client.get_collections().collections]

        if COLLECTION_NAME in existing:
            info = self.client.get_collection(COLLECTION_NAME)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict) and "dense" in vectors_config:
                return
            logger.warning(f"Recreating {COLLECTION_NAME} with named vectors")
            self.client.delete_collection(COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=self.dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )

        # Create payload index for filtered search
        self.client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="note_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        logger.info(f"Created collection {COLLECTION_NAME} with note_type index")

    def _discover_notes(self) -> List[Path]:
        """Find all .md files in the vault, excluding template/meta dirs."""
        notes = []
        for md_file in self.vault_path.rglob("*.md"):
            # Skip excluded directories
            rel = md_file.relative_to(self.vault_path)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            notes.append(md_file)
        return notes

    def _parse_note(self, path: Path) -> Optional[Dict]:
        """Parse a vault note: extract frontmatter, content, wiki-links."""
        try:
            post = frontmatter.load(str(path))
        except Exception as e:
            logger.warning(f"Failed to parse {path}: {e}")
            return None

        content = post.content
        metadata = dict(post.metadata)

        # Extract wiki-links
        wiki_links = WIKILINK_RE.findall(content)

        # Relative path for identification
        rel_path = str(path.relative_to(self.vault_path))

        # Determine note type from frontmatter or directory
        note_type = metadata.get("type", "")
        if not note_type:
            parts = path.relative_to(self.vault_path).parts
            if len(parts) > 1:
                dir_to_type = {
                    "competitors": "competitor",
                    "brands": "brand",
                    "categories": "category",
                    "market": "market",
                    "translations": "translation",
                    "strategy": "strategy",
                    "recommendations": "recommendation",
                }
                note_type = dir_to_type.get(parts[0], "other")

        # Get title from first heading or filename
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else path.stem.replace("-", " ").title()

        return {
            "path": rel_path,
            "title": title,
            "note_type": note_type,
            "content": content,
            "frontmatter": metadata,
            "wiki_links": wiki_links,
        }

    async def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def _has_changed(self, note_path: str, content: str) -> bool:
        """Check if note content has changed since last index."""
        if not self.redis:
            return True
        new_hash = await self._compute_hash(content)
        redis_key = f"{HASH_PREFIX}:{note_path}"
        old_hash = await self.redis.get(redis_key)
        if old_hash:
            old_hash = old_hash.decode() if isinstance(old_hash, bytes) else old_hash
        return old_hash != new_hash

    async def _set_hash(self, note_path: str, content: str):
        """Store content hash after successful indexing."""
        if not self.redis:
            return
        content_hash = await self._compute_hash(content)
        redis_key = f"{HASH_PREFIX}:{note_path}"
        await self.redis.set(redis_key, content_hash)

    async def index_vault(self, force: bool = False) -> Dict:
        """Index all vault notes into Qdrant.

        Args:
            force: If True, re-index all notes regardless of content hash.

        Returns:
            Dict with indexed/skipped/errors counts.
        """
        from sparse_encoder import encode_sparse

        self._ensure_collection()

        notes = self._discover_notes()
        logger.info(f"Discovered {len(notes)} vault notes")

        indexed = 0
        skipped = 0
        errors = 0
        indexed_paths: Set[str] = set()

        loop = asyncio.get_event_loop()

        for note_path in notes:
            try:
                parsed = self._parse_note(note_path)
                if not parsed:
                    errors += 1
                    continue

                rel_path = parsed["path"]
                indexed_paths.add(rel_path)

                # Content hash dedup
                if not force and not await self._has_changed(rel_path, parsed["content"]):
                    skipped += 1
                    continue

                # Chunk the note
                chunks = _chunk_markdown(parsed["content"])
                if not chunks:
                    skipped += 1
                    continue

                # Prepare texts for embedding
                texts = [
                    f"{parsed['title']} - {chunk['section_heading']}\n{chunk['text']}"
                    for chunk in chunks
                ]

                # Embed via BGE-M3
                dense_embeddings, sparse_embeddings = await loop.run_in_executor(
                    None,
                    lambda t=texts: bgem3_encoder.encode_both(t)
                )

                # Build points
                points = []
                for i, chunk in enumerate(chunks):
                    point_id = _generate_point_id(rel_path, i)
                    payload = {
                        "text": chunk["text"],
                        "note_type": parsed["note_type"],
                        "note_path": rel_path,
                        "note_title": parsed["title"],
                        "section_heading": chunk["section_heading"],
                        "chunk_index": chunk["chunk_index"],
                        "wiki_links": parsed["wiki_links"],
                        "frontmatter": parsed["frontmatter"],
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "source_type": "vault",
                    }
                    points.append(PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_embeddings[i].tolist(),
                            "sparse": sparse_embeddings[i],
                        },
                        payload=payload,
                    ))

                # Upsert to Qdrant
                if points:
                    await loop.run_in_executor(
                        None,
                        lambda p=points: self.client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=p,
                        )
                    )

                # Update content hash
                await self._set_hash(rel_path, parsed["content"])
                indexed += 1

            except Exception as e:
                logger.error(f"Failed to index {note_path}: {e}")
                errors += 1

        # Clean up orphaned points (notes that were deleted from vault)
        orphans_deleted = await self._delete_orphans(indexed_paths)

        result = {
            "indexed": indexed,
            "skipped": skipped,
            "errors": errors,
            "orphans_deleted": orphans_deleted,
            "total_notes": len(notes),
        }
        logger.info(f"Vault indexing complete: {result}")
        return result

    async def _delete_orphans(self, current_paths: Set[str]) -> int:
        """Delete Qdrant points for notes that no longer exist in the vault."""
        try:
            # Scroll through all points to find orphaned note_paths
            all_paths_in_qdrant: Set[str] = set()
            offset = None

            while True:
                result = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=100,
                    offset=offset,
                    with_payload=["note_path"],
                )
                points, next_offset = result
                for point in points:
                    path = point.payload.get("note_path", "")
                    if path:
                        all_paths_in_qdrant.add(path)

                if next_offset is None:
                    break
                offset = next_offset

            orphan_paths = all_paths_in_qdrant - current_paths
            if not orphan_paths:
                return 0

            # Delete points with orphaned paths
            deleted = 0
            for path in orphan_paths:
                self.client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=Filter(
                        must=[FieldCondition(key="note_path", match=MatchValue(value=path))]
                    ),
                )
                # Clean up Redis hash
                if self.redis:
                    await self.redis.delete(f"{HASH_PREFIX}:{path}")
                deleted += 1
                logger.info(f"Deleted orphaned note from index: {path}")

            return deleted

        except Exception as e:
            logger.warning(f"Orphan cleanup failed: {e}")
            return 0

    def get_collection_info(self) -> Optional[Dict]:
        """Get knowledge_brain collection stats."""
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            if COLLECTION_NAME not in existing:
                return None

            info = self.client.get_collection(COLLECTION_NAME)
            return {
                "collection": COLLECTION_NAME,
                "points_count": info.points_count,
                "vectors_count": getattr(info, "vectors_count", info.points_count),
                "status": info.status.value if info.status else "unknown",
            }
        except Exception as e:
            logger.warning(f"Failed to get collection info: {e}")
            return None

    def search_recommendations(
        self,
        query: str,
        top_k: int = 5,
        platform_filter: str = None,
        note_type: str = "recommendation",
    ) -> list[dict]:
        """Hybrid search on knowledge_brain filtered by note_type."""
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            if COLLECTION_NAME not in existing:
                return []

            dense_embedding = bgem3_encoder.encode_dense_query(query)
            sparse_embedding = bgem3_encoder.encode_sparse_query(query)

            conditions = [
                FieldCondition(key="note_type", match=MatchValue(value=note_type))
            ]
            if platform_filter:
                conditions.append(
                    FieldCondition(
                        key="frontmatter.platform",
                        match=MatchValue(value=platform_filter),
                    )
                )

            search_filter = Filter(must=conditions)

            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    Prefetch(query=dense_embedding, using="dense", filter=search_filter, limit=top_k * 3),
                    Prefetch(query=sparse_embedding, using="sparse", filter=search_filter, limit=top_k * 3),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

            return [
                {
                    "title": r.payload.get("note_title", ""),
                    "section_heading": r.payload.get("section_heading", ""),
                    "text": r.payload.get("text", ""),
                    "note_path": r.payload.get("note_path", ""),
                    "platform": (r.payload.get("frontmatter") or {}).get("platform", ""),
                    "confidence": (r.payload.get("frontmatter") or {}).get("confidence", ""),
                    "handle": (r.payload.get("frontmatter") or {}).get("handle", ""),
                    "score": round(r.score, 4),
                    "source_type": "recommendation",
                }
                for r in results.points
            ]

        except Exception as e:
            logger.error(f"Recommendation search error: {e}")
            return []
