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
    _collection = _client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def upsert_movies(movies: list[dict]):
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
            "similarity": 1 - dist,
        })
    return out


def count() -> int:
    return get_collection().count()


def ensure_index_loaded():
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
