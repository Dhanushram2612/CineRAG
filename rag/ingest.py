"""
Builds the base catalog from the static Kaggle "tmdb-movie-metadata" CSV
export — a one-time file download, not a live API call. This project uses
OMDb (not TMDB) for everything live, since TMDB's API/website is blocked on
some networks (notably bundled into anti-piracy blocklists on parts of the
Indian network) — so catalog building can't depend on it either.

Produces a parquet snapshot loaded into ChromaDB. The deployed app just
loads the snapshot; it never has to re-embed the whole catalog live.

Download the dataset from Kaggle: search "tmdb-movie-metadata"
(tmdb_5000_movies.csv is required; tmdb_5000_credits.csv is optional but
recommended — it adds director + cast to the embedded text, which noticeably
improves match quality over title+genre+overview alone).

Usage:
    python -m rag.ingest --movies tmdb_5000_movies.csv --credits tmdb_5000_credits.csv
    python -m rag.ingest --movies tmdb_5000_movies.csv   # credits optional
"""
import argparse
import ast
import pandas as pd
from . import config, vectorstore


def _safe_eval(x):
    if pd.isna(x) or x == "":
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def _genre_names(x) -> str:
    items = _safe_eval(x)
    names = [i.get("name", "") for i in items if isinstance(i, dict) and i.get("name")]
    return ", ".join(names)


def _cast_names(x, limit: int = 5) -> str:
    items = _safe_eval(x)
    names = [i.get("name", "") for i in items if isinstance(i, dict) and i.get("name")]
    return ", ".join(names[:limit])


def _director_name(crew_x) -> str:
    return _crew_job_name(crew_x, {"Director"})


def _crew_job_name(crew_x, jobs: set[str]) -> str:
    """Generic version of _director_name — finds the first crew member
    whose job matches any of the given job titles. Used for composer and
    writer too, since TMDB's crew job field uses several different labels
    for what's conceptually the same role (e.g. a writer might be credited
    as "Writer", "Screenplay", or "Story")."""
    items = _safe_eval(crew_x)
    for member in items:
        if isinstance(member, dict) and member.get("job") in jobs:
            return member.get("name", "")
    return ""


def _composer_name(crew_x) -> str:
    return _crew_job_name(crew_x, {"Original Music Composer", "Music"})


def _writer_name(crew_x) -> str:
    # Checked in this priority order: screenplay is the most direct writing
    # credit; "Story" is used when someone's credited for the underlying
    # story but not the screenplay itself.
    return (_crew_job_name(crew_x, {"Screenplay"})
            or _crew_job_name(crew_x, {"Writer"})
            or _crew_job_name(crew_x, {"Story"}))


def build_combined_text(m: dict) -> str:
    genres = m.get("genres", "")
    overview = m.get("overview") or "No overview available."
    parts = [f"{m['title']}. Genres: {genres}. {overview}"]
    if m.get("director"):
        parts.append(f"Directed by {m['director']}.")
    if m.get("writer"):
        parts.append(f"Written by {m['writer']}.")
    if m.get("composer"):
        parts.append(f"Music by {m['composer']}.")
    if m.get("cast"):
        parts.append(f"Starring {m['cast']}.")
    return " ".join(parts)


def normalize_row(row: pd.Series) -> dict:
    movie = {
        "id": int(row["id"]),
        "title": row.get("title") or row.get("original_title") or "Untitled",
        "genres": _genre_names(row.get("genres")),
        "overview": row.get("overview") or "",
        "vote_average": float(row.get("vote_average") or 0),
        "vote_count": int(row.get("vote_count") or 0),
        "release_date": row.get("release_date", "") or "",
        "language": row.get("original_language") or "en",
        "director": row.get("director", "") or "",
        "cast": row.get("cast_names", "") or "",
        "composer": row.get("composer_name", "") or "",
        "writer": row.get("writer_name", "") or "",
    }
    movie["combined_text"] = build_combined_text(movie)
    return movie


def run(movies_csv: str, credits_csv: str | None = None):
    df = pd.read_csv(movies_csv)

    if credits_csv:
        credits = pd.read_csv(credits_csv).rename(columns={"movie_id": "id"})
        credits["director"] = credits["crew"].apply(_director_name)
        credits["composer_name"] = credits["crew"].apply(_composer_name)
        credits["writer_name"] = credits["crew"].apply(_writer_name)
        credits["cast_names"] = credits["cast"].apply(_cast_names)
        df = df.merge(
            credits[["id", "director", "composer_name", "writer_name", "cast_names"]],
            on="id", how="left",
        )

    records = [normalize_row(row) for _, row in df.iterrows()]
    records = [r for r in records if len(r["combined_text"]) > 20]

    if not records:
        raise RuntimeError(f"No usable rows found in {movies_csv}")

    out_df = pd.DataFrame(records)
    out_df.to_parquet(config.SNAPSHOT_PATH, index=False)
    print(f"[ingest] snapshot saved: {config.SNAPSHOT_PATH} ({len(out_df)} rows)")
    if credits_csv:
        with_director = sum(1 for r in records if r["director"])
        with_composer = sum(1 for r in records if r["composer"])
        with_writer = sum(1 for r in records if r["writer"])
        print(f"[ingest] {with_director}/{len(records)} movies matched to a director via credits.csv")
        print(f"[ingest] {with_composer}/{len(records)} movies matched to a composer via credits.csv")
        print(f"[ingest] {with_writer}/{len(records)} movies matched to a writer via credits.csv")

    vectorstore.upsert_movies(records)
    print(f"[ingest] chroma collection now has {vectorstore.count()} movies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--movies", required=True, help="path to tmdb_5000_movies.csv")
    parser.add_argument("--credits", default=None, help="path to tmdb_5000_credits.csv (optional)")
    args = parser.parse_args()
    run(args.movies, args.credits)
