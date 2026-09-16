
from . import config, tmdb_client, omdb_client


def _enrich_tmdb(candidates: list[dict]) -> dict[int, dict]:
    ids = [c["id"] for c in candidates]
    raw = tmdb_client.enrich_many(ids)
    out = {}
    for mid, detail in raw.items():
        out[mid] = {
            "vote_average": detail.get("vote_average"),
            "overview": detail.get("overview"),
            "poster_url": tmdb_client.poster_url(detail.get("poster_path")),
        }
    return out


def enrich_many(candidates: list[dict]) -> dict[int, dict]:
    """candidates: retrieved movies, each with 'id' and 'title' at minimum."""
    backend = config.ENRICHMENT_BACKEND
    if backend == "omdb":
        return omdb_client.enrich_many(candidates)
    if backend == "tmdb":
        return _enrich_tmdb(candidates)
    raise ValueError(f"Unknown ENRICHMENT_BACKEND '{backend}' — use 'omdb' or 'tmdb'.")
