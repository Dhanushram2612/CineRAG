"""
ChromaDB wrapper. Chroma handles cosine similarity, metadata filtering, and
persistence natively — this is what replaces the raw FAISS + manual
re-encoding-per-query approach from v1.
"""
import chromadb
from chromadb.utils import embedding_functions
from . import config

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )
    # cosine space explicitly — the #1 fix from the v1 audit (raw L2 on
    # un-normalized sentence embeddings biases results toward vector magnitude)
    _collection = _client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def upsert_movies(movies: list[dict]):
    """movies: list of dicts with id, title, genres, overview, vote_average,
    vote_count, release_date, language, combined_text. director/cast/
    composer/writer are optional (present when ingest.py was run with
    --credits)."""
    if not movies:
        return
    col = get_collection()
    col.upsert(
        ids=[str(m["id"]) for m in movies],
        documents=[m["combined_text"] for m in movies],
        metadatas=[
            {
                "title": m["title"],
                "genres": m["genres"],
                "vote_average": m["vote_average"],
                "vote_count": m["vote_count"],
                "release_date": m.get("release_date", ""),
                "language": m.get("language", "en"),
                "director": m.get("director", ""),
                "cast": m.get("cast", ""),
                "composer": m.get("composer", ""),
                "writer": m.get("writer", ""),
            }
            for m in movies
        ],
    )


def search(query_text: str, n_results: int, language: str | None = None,
           min_vote_count: int = 0, min_vote_average: float = 0,
           director: str | None = None, cast_contains: str | None = None,
           composer: str | None = None, writer: str | None = None) -> list[dict]:
    """Semantic search with optional metadata filters applied by Chroma
    itself (no manual re-encoding of a filtered subset, unlike v1).

    min_vote_average: filters out movies below this rating outright, even if
    they're a strong semantic match. Similarity alone doesn't mean quality —
    a thematically perfect but genuinely bad movie erodes user trust more
    than an honest "no strong match" would.

    director, composer, writer: exact match on the corresponding metadata
    field — each holds a single name, so unlike cast this can use Chroma's
    `where` directly. Semantic search alone is weak at exact-name recall (it
    ranks by theme/meaning, not entity identity) — a query like "hans
    zimmer music" can miss his actual films if they don't rank in the top-N
    purely on similarity. Passing a resolved name here filters to exactly
    that person's films first, THEN ranks those semantically.

    cast_contains: substring match against the embedded document text
    (which includes "Starring <cast>"), NOT a metadata equality filter —
    the 'cast' metadata field holds multiple names joined into one string
    per movie ("Actor1, Actor2, ..."), so an exact match against it would
    only ever match if the query equaled that entire joined string. Uses
    Chroma's where_document instead of where for this reason."""
    col = get_collection()

    conditions = []
    if language:
        conditions.append({"language": language})
    if min_vote_count:
        conditions.append({"vote_count": {"$gte": min_vote_count}})
    if min_vote_average:
        conditions.append({"vote_average": {"$gte": min_vote_average}})
    if director:
        conditions.append({"director": director})
    if composer:
        conditions.append({"composer": composer})
    if writer:
        conditions.append({"writer": writer})

    # Chroma requires a single condition as a plain dict, but rejects a flat
    # multi-key dict for combining more than one ("Expected where to have
    # exactly one operator") — multiple conditions must be wrapped in $and.
    if len(conditions) == 0:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    where_document = {"$contains": cast_contains} if cast_contains else None

    results = col.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
        where_document=where_document,
    )

    out = []
    ids = results.get("ids", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for mid, meta, doc, dist in zip(ids, metas, docs, dists):
        out.append({
            "id": int(mid),
            "title": meta.get("title"),
            "genres": meta.get("genres"),
            "vote_average": meta.get("vote_average"),
            "vote_count": meta.get("vote_count"),
            "director": meta.get("director", ""),
            "cast": meta.get("cast", ""),
            "composer": meta.get("composer", ""),
            "writer": meta.get("writer", ""),
            "overview_snippet": doc,
            "similarity": 1 - dist,  # cosine distance -> similarity
        })
    return out


def count() -> int:
    return get_collection().count()


def ensure_index_loaded():
    """On a fresh deploy the Chroma dir may be empty even though a snapshot
    parquet exists in the repo. Load from the snapshot instead of forcing a
    live TMDB re-fetch of the whole catalog on cold start."""
    import os
    import pandas as pd
    from . import config

    if count() > 0:
        return

    if not os.path.exists(config.SNAPSHOT_PATH):
        print("[vectorstore] no snapshot found — run `python -m rag.ingest` first.")
        return

    df = pd.read_parquet(config.SNAPSHOT_PATH)
    upsert_movies(df.to_dict("records"))
    print(f"[vectorstore] loaded {len(df)} movies from snapshot into Chroma")
