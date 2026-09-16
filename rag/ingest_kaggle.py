
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


def build_combined_text(m: dict) -> str:
    """Matches ingest.py's build_combined_text exactly — same schema either
    ingest path produces, so retrieval quality doesn't depend on which one
    built the catalog."""
    genres = m.get("genres", "")
    overview = m.get("overview") or "No overview available."
    return f"{m['title']}. Genres: {genres}. {overview}"


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
    }
    movie["combined_text"] = build_combined_text(movie)
    return movie


def run(movies_csv: str):
    df = pd.read_csv(movies_csv)
    records = [normalize_row(row) for _, row in df.iterrows()]
    records = [r for r in records if len(r["combined_text"]) > 20]

    if not records:
        raise RuntimeError(f"No usable rows found in {movies_csv}")

    out_df = pd.DataFrame(records)
    out_df.to_parquet(config.SNAPSHOT_PATH, index=False)
    print(f"[ingest_kaggle] snapshot saved: {config.SNAPSHOT_PATH} ({len(out_df)} rows)")

    vectorstore.upsert_movies(records)
    print(f"[ingest_kaggle] chroma collection now has {vectorstore.count()} movies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--movies", required=True, help="path to tmdb_5000_movies.csv")
    args = parser.parse_args()
    run(args.movies)
