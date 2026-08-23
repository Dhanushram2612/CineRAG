"""
Central configuration. Every secret comes from the environment (.env locally,
Space secrets on HF, Streamlit secrets on Streamlit Cloud) — never hardcoded.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root explicitly (the folder containing app.py,
# one level up from this file inside rag/) — not just "wherever the current
# working directory happens to be." This avoids the common gotcha where
# `.env` exists but isn't found because the app was launched from a
# different directory than expected.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# --- LLM backend switch ---
# "ollama" for local dev, "groq" for cloud deployment. Same interface either way.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Live enrichment (poster/rating refresh on retrieved results) ---
# OMDb only — TMDB's API/website is blocked on some networks (some Indian
# ISPs), so the whole data layer avoids depending on it.
OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "")

# --- Vector store ---
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./data/chroma")
COLLECTION_NAME = "movies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K = 8
MIN_VOTE_COUNT = 50  # filters out movies with unreliable/near-zero-vote ratings
# Filters out movies below this average rating (out of 10), even if they'd
# otherwise rank as a strong semantic match. Semantic similarity alone says
# nothing about whether a movie is actually good — recommending something
# that matches the theme but has a 3/10 rating erodes trust in the whole
# system faster than an occasional weak/no match does. 6.0 is a reasonable
# starting floor ("above average, not necessarily great") — raise it for a
# stricter bar, lower it if genuinely good niche/cult films are getting
# excluded by too aggressive a cutoff.
MIN_VOTE_AVERAGE = float(os.environ.get("MIN_VOTE_AVERAGE", "6.0"))

# --- Data snapshot (cached corpus for fast cold starts) ---
SNAPSHOT_PATH = os.environ.get("SNAPSHOT_PATH", "./data/movies_snapshot.parquet")


def validate():
    """Call at startup so missing secrets fail loudly, not three layers deep."""
    missing = []
    if not OMDB_API_KEY:
        missing.append("OMDB_API_KEY")
    if LLM_BACKEND == "groq" and not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Set them in a .env file locally or in your host's secrets manager. "
            "Never hardcode them in source."
        )
