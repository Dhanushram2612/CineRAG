"""
OMDb API client — live enrichment for retrieved movies (poster, rating,
plot). Chosen over TMDB because TMDB's API/website is blocked on some
networks (some Indian ISPs bundle themoviedb.org into anti-piracy
blocklists). OMDb is a different provider/domain and unaffected.

OMDb has no bulk "discover" endpoint, only title lookup — fine here since
we're only enriching an already-retrieved shortlist (~5-8 movies/query),
not building a catalog from scratch (see ingest.py for that).

Free tier: 1,000 requests/day via https://www.omdbapi.com/apikey.aspx
"""
import time
import requests
from . import config

_session = requests.Session()
BASE_URL = "https://www.omdbapi.com/"


def _get(params: dict, retries: int = 2) -> dict | None:
    """Best-effort — enrichment should never crash the whole recommendation
    response over one flaky lookup."""
    params = dict(params)
    params["apikey"] = config.OMDB_API_KEY

    for attempt in range(retries):
        try:
            resp = _session.get(BASE_URL, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            if attempt == retries - 1:
                return None
            time.sleep(1)
            continue

        if data.get("Response") != "True":
            return None
        return data

    return None


def fetch_by_title(title: str, year: str | None = None) -> dict | None:
    params = {"t": title, "type": "movie"}
    if year:
        params["y"] = year
    return _get(params)


def poster_url(omdb_data: dict) -> str | None:
    poster = omdb_data.get("Poster")
    if not poster or poster == "N/A":
        return None
    return poster


def vote_average(omdb_data: dict) -> float | None:
    rating = omdb_data.get("imdbRating")
    if not rating or rating == "N/A":
        return None
    try:
        return float(rating)
    except ValueError:
        return None


def enrich_many(candidates: list[dict]) -> dict[int, dict]:
    """candidates: retrieved movies with at least 'id' and 'title'
    ('release_date' optional, used to disambiguate same-titled movies).

    Returns {id: {"vote_average", "overview", "poster_url"}}."""
    out = {}
    for c in candidates:
        year = (c.get("release_date") or "")[:4] or None
        data = fetch_by_title(c["title"], year=year)
        if not data:
            print(f"[omdb] no match for '{c['title']}' ({year})")
            continue
        out[c["id"]] = {
            "vote_average": vote_average(data),
            "overview": data.get("Plot") if data.get("Plot") not in (None, "N/A") else None,
            "poster_url": poster_url(data),
        }
    return out
