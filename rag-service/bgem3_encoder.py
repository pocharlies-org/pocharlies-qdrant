"""
BGE-M3 encoder via remote TEI (Text Embeddings Inference) service.
Replaces local FlagEmbedding model with HTTP calls to GPU-accelerated TEI.

Usage:
    from bgem3_encoder import encode_dense, encode_dense_query, encode_sparse, encode_sparse_query, encode_both
"""

import logging
import os
import re
from typing import List, Tuple

import numpy as np
import httpx
from qdrant_client.http.models import SparseVector

logger = logging.getLogger(__name__)

DENSE_DIM = 1024

# TEI endpoint — set via env or default to localhost:8090
TEI_URL = os.getenv("TEI_EMBEDDING_URL", "http://localhost:8090").rstrip("/")

# Reusable sync client with connection pooling
_client = httpx.Client(timeout=120.0, limits=httpx.Limits(max_connections=10))


def _tei_dense(texts: List[str]) -> List[List[float]]:
    """Call TEI /embed endpoint for dense vectors."""
    resp = _client.post(f"{TEI_URL}/embed", json={"inputs": texts})
    resp.raise_for_status()
    return resp.json()


def _tei_sparse(texts: List[str]) -> List[List[dict]]:
    """Call TEI /embed_sparse endpoint for sparse vectors."""
    resp = _client.post(f"{TEI_URL}/embed_sparse", json={"inputs": texts})
    resp.raise_for_status()
    return resp.json()


def _sparse_dicts_to_qdrant(sparse_list: List[dict]) -> SparseVector:
    """Convert TEI sparse response [{"index":N,"value":V},...] to Qdrant SparseVector."""
    if not sparse_list:
        return SparseVector(indices=[], values=[])
    indices = [entry["index"] for entry in sparse_list]
    values = [float(entry["value"]) for entry in sparse_list]
    return SparseVector(indices=indices, values=values)


def encode_dense(texts: List[str]) -> np.ndarray:
    """Batch encode texts into 1024d dense vectors."""
    vecs = _tei_dense(texts)
    return np.array(vecs, dtype=np.float32)


def encode_dense_query(text: str) -> List[float]:
    """Encode a single query into a 1024d dense vector."""
    vecs = _tei_dense([text])
    return vecs[0]


def encode_sparse(texts: List[str]) -> List[SparseVector]:
    """Batch encode texts into sparse vectors via TEI."""
    raw = _tei_sparse(texts)
    return [_sparse_dicts_to_qdrant(entry) for entry in raw]


def encode_sparse_query(text: str) -> SparseVector:
    """Encode query with English enrichment for cross-lingual sparse matching."""
    enriched = _enrich_query_english(text)
    raw = _tei_sparse([enriched])
    return _sparse_dicts_to_qdrant(raw[0])


def encode_both(texts: List[str]) -> Tuple[np.ndarray, List[SparseVector]]:
    """Encode texts returning both dense and sparse (two HTTP calls)."""
    dense = encode_dense(texts)
    sparse = encode_sparse(texts)
    return dense, sparse


# ── Bilingual airsoft glossary for sparse query enrichment ──────────

AIRSOFT_TRANSLATIONS = [
    # Spanish — multi-word first
    (re.compile(r'\bca[ñn]o(?:n|nes)?\s*interno', re.I), 'inner barrel precision'),
    (re.compile(r'\bca[ñn]o(?:n|nes)?\s*externo', re.I), 'outer barrel'),
    (re.compile(r'\bca[ñn]o(?:n|nes)?', re.I), 'barrel'),
    (re.compile(r'\bmuelle(?:s)?', re.I), 'spring'),
    (re.compile(r'\btolva', re.I), 'hop-up'),
    (re.compile(r'\bcargador(?:es)?', re.I), 'magazine'),
    (re.compile(r'\bculata', re.I), 'stock buttstock'),
    (re.compile(r'\bgatillo', re.I), 'trigger'),
    (re.compile(r'\bmira\b', re.I), 'scope sight optic'),
    (re.compile(r'\bsilenciador', re.I), 'suppressor silencer'),
    (re.compile(r'\bbater[ií]a', re.I), 'battery'),
    (re.compile(r'\bbolas?\b', re.I), 'BBs ammunition'),
    (re.compile(r'\bmejorar?\b', re.I), 'upgrade improve'),
    (re.compile(r'\brepuesto(?:s)?', re.I), 'spare parts replacement'),
    (re.compile(r'\bfrancotirador', re.I), 'sniper'),
    (re.compile(r'\bfusil', re.I), 'rifle'),
    (re.compile(r'\bpistola', re.I), 'pistol handgun'),
    (re.compile(r'\bgafas?\b', re.I), 'goggles eye protection'),
    (re.compile(r'\bchaleco', re.I), 'plate carrier vest'),
    (re.compile(r'\bcorrea', re.I), 'sling'),
    (re.compile(r'\bfunda', re.I), 'holster case'),
    (re.compile(r'\bempuñadura', re.I), 'grip'),
    (re.compile(r'\bguardamanos', re.I), 'handguard rail'),
    (re.compile(r'\bprecisi[oó]n', re.I), 'precision accuracy'),
    (re.compile(r'\baccesorio(?:s)?', re.I), 'accessories'),
    (re.compile(r'\bbip[oó]de', re.I), 'bipod'),
    (re.compile(r'\bselector', re.I), 'selector switch'),
    (re.compile(r'\bmascara', re.I), 'mask face protection'),
    # French
    (re.compile(r'\bcanon\s*interne', re.I), 'inner barrel precision'),
    (re.compile(r'\bcanon\b', re.I), 'barrel'),
    (re.compile(r'\bressort', re.I), 'spring'),
    (re.compile(r'\bchargeur', re.I), 'magazine'),
    (re.compile(r'\blunette', re.I), 'scope optic'),
    (re.compile(r'\bam[eé]liorer', re.I), 'upgrade improve'),
    # German
    (re.compile(r'\binnenlauf', re.I), 'inner barrel'),
    (re.compile(r'\blauf\b', re.I), 'barrel'),
    (re.compile(r'\bfeder\b', re.I), 'spring'),
    (re.compile(r'\bmagazin\b', re.I), 'magazine'),
    (re.compile(r'\bzielfernrohr', re.I), 'scope optic'),
    (re.compile(r'\bverbesser', re.I), 'upgrade improve'),
]


def _enrich_query_english(query: str) -> str:
    """Append English keywords for non-English airsoft terms found in query."""
    additions = []
    for pattern, english in AIRSOFT_TRANSLATIONS:
        if pattern.search(query):
            additions.append(english)
    if not additions:
        return query
    enriched = f"{query} {' '.join(additions)}"
    logger.debug("Sparse enrichment: '%s' → '%s'", query, enriched)
    return enriched
