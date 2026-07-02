"""Vector index — the Qdrant collection that holds chunk embeddings + payloads.

WHY Qdrant (vs. the embedded Chroma the original demo used): an enterprise engine wants
a real ANN server you can point multiple processes at, inspect over HTTP, and tune. The
index is an HNSW graph (see config.py for the M / ef_construct / ef_search knobs).

This module owns:
  * creating the collection with the right vector size + HNSW params,
  * upserting chunks (vector + payload), and
  * the low-level dense search retrieve.py calls.

A "payload" is the metadata stored alongside each vector (project_id, source_set, text,
etc.). It's what lets retrieval return citations AND lets us filter by facet (e.g.
source_set == "atlas") at search time — the metadata-filter half of the two-query-class
design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .chunking import Chunk
from .config import SETTINGS, Settings

# Every Qdrant call below resolves its collection name from `settings.collection_name`
# (model-derived unless RAGEVAL_COLLECTION overrides). Because ingest and retrieve share the
# same Settings, they always agree on which collection — the index a model wrote is the one
# it reads. NO module-level COLLECTION_NAME constant: that would silently couple all models
# to one index and break the A/B.


def get_client(settings: Settings = SETTINGS) -> QdrantClient:
    """Open a Qdrant client against the configured URL (docker-compose.yml)."""
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient, settings: Settings = SETTINGS,
                      *, recreate: bool = False) -> None:
    """Create the collection if needed, with explicit HNSW + vector params.

    The vector size MUST equal the embedding model's dimensionality, and the distance
    must be COSINE to match our unit-normalised embeddings. We set HNSW M / ef_construct
    here (build-time graph quality); ef_search is set per-query in retrieve.py.
    """
    name = settings.collection_name
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=settings.embed_dim,
                distance=qm.Distance.COSINE,
            ),
            hnsw_config=qm.HnswConfigDiff(
                m=settings.hnsw_m,
                ef_construct=settings.hnsw_ef_construct,
            ),
        )
        # Payload indexes make metadata FILTERS fast (the aggregation/facet path). We
        # index the fields we filter/group on most.
        for field in ("project_id", "source_set", "doc_type"):
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )


def _point_id(chunk_id: str) -> str:
    """Qdrant point ids must be uint or UUID. We derive a stable UUID5 from the chunk's
    string id so re-ingesting the same chunk upserts (replaces) rather than duplicates."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


# Qdrant rejects a single HTTP request whose body exceeds ~32 MB. Each point is a 384-d
# vector + a text payload (~12 KB), so a full real-corpus ingest (tens of thousands of
# chunks) is hundreds of MB in one call. Upload in batches to stay well under the limit;
# 256 points ≈ 3 MB per request.
_UPSERT_BATCH = 256


def upsert_chunks(client: QdrantClient, chunks: list[Chunk],
                  vectors: list[list[float]], settings: Settings = SETTINGS) -> int:
    """Insert-or-replace chunks with their vectors and payloads, in batches. Returns the count."""
    points = []
    for c, v in zip(chunks, vectors):
        points.append(
            qm.PointStruct(
                id=_point_id(c.id),
                vector=v,
                payload={
                    "chunk_id": c.id,
                    "project_id": c.project_id,
                    "source_set": c.source_set,
                    "source": c.source,
                    "doc_type": c.doc_type,
                    "chunk_index": c.chunk_index,
                    "page": c.page,          # PDF page provenance (None for non-PDF chunks)
                    "text": c.text,
                },
            )
        )
    n_batches = (len(points) + _UPSERT_BATCH - 1) // _UPSERT_BATCH
    for k, start in enumerate(range(0, len(points), _UPSERT_BATCH), 1):
        client.upsert(
            collection_name=settings.collection_name,
            points=points[start:start + _UPSERT_BATCH],
        )
        # FLUSHED per-batch progress so a long, redirected upsert shows live (not at the end).
        print(f"[upsert] batch {k}/{n_batches} ({min(start + _UPSERT_BATCH, len(points))}"
              f"/{len(points)} points)", flush=True)
    return len(points)


@dataclass(frozen=True)
class DenseHit:
    """One dense (vector) search result."""
    chunk_id: str
    text: str
    project_id: str
    source_set: str
    source: str
    doc_type: str
    chunk_index: int
    score: float  # cosine similarity in [0, 1]
    page: int | None = None  # PDF page provenance (None for non-PDF chunks)


def dense_search(client: QdrantClient, query_vector: list[float], *, limit: int,
                 ef_search: int, source_set: str | None = None,
                 settings: Settings = SETTINGS) -> list[DenseHit]:
    """Dense ANN search over the HNSW index, optionally filtered by source_set.

    `ef_search` is the per-query candidate-list size — higher = better recall, slower.
    The optional `source_set` filter is the metadata-filter path (e.g. restrict to one
    source-set for a set-intersection question).
    """
    flt = None
    if source_set is not None:
        flt = qm.Filter(must=[qm.FieldCondition(key="source_set",
                                                match=qm.MatchValue(value=source_set))])
    res = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
        query_filter=flt,
        search_params=qm.SearchParams(hnsw_ef=ef_search),
    ).points
    hits: list[DenseHit] = []
    for p in res:
        pl = p.payload or {}
        hits.append(
            DenseHit(
                chunk_id=str(pl.get("chunk_id", "")),
                text=str(pl.get("text", "")),
                project_id=str(pl.get("project_id", "")),
                source_set=str(pl.get("source_set", "")),
                source=str(pl.get("source", "")),
                doc_type=str(pl.get("doc_type", "")),
                chunk_index=int(pl.get("chunk_index", -1)),
                score=float(p.score),  # cosine distance config → score is similarity
                page=(int(pl["page"]) if pl.get("page") is not None else None),
            )
        )
    return hits


def scroll_all(client: QdrantClient, *, project_id: str | None = None,
               settings: Settings = SETTINGS) -> Iterable[dict]:
    """Iterate every stored chunk's payload (optionally for one project).

    This powers the chunk inspector: structured `scroll()` over the collection to eyeball
    exactly what became an embedding (boilerplate, truncated .docx, leaked code, etc.).
    """
    flt = None
    if project_id is not None:
        flt = qm.Filter(must=[qm.FieldCondition(key="project_id",
                                                match=qm.MatchValue(value=project_id))])
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.collection_name,
            scroll_filter=flt,
            with_payload=True,
            limit=256,
            offset=offset,
        )
        for p in points:
            yield p.payload or {}
        if offset is None:
            break


def count(client: QdrantClient, settings: Settings = SETTINGS) -> int:
    """Total stored chunks (points) in the collection."""
    name = settings.collection_name
    if not client.collection_exists(name):
        return 0
    return client.count(collection_name=name).count


def existing_source_sets(client: QdrantClient, settings: Settings = SETTINGS,
                         *, sample_points: int = 256) -> set[str]:
    """The distinct `source_set` values already stored in the target collection (sampled).

    Used by the ingest contamination guard (defense-in-depth): before upserting, peek at what's
    already in the collection so we can WARN if a sample ingest is about to write into a collection
    that holds real-corpus points, or vice versa. Reads at most `sample_points` payloads — enough
    to see the families present without scanning a 12k-point collection. Empty set when the
    collection is absent or empty (the normal first-ingest case)."""
    name = settings.collection_name
    if not client.collection_exists(name):
        return set()
    points, _ = client.scroll(collection_name=name, with_payload=["source_set"],
                              limit=sample_points)
    return {str((p.payload or {}).get("source_set", "")) for p in points if p.payload}
