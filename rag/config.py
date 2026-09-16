import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "")

CHROMA_DIR = os.environ.get("CHROMA_DIR", "./data/chroma")
COLLECTION_NAME = "movies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


TOP_K = 20
MIN_VOTE_COUNT = 50 
MIN_VOTE_AVERAGE = float(os.environ.get("MIN_VOTE_AVERAGE", "6.0"))

SNAPSHOT_PATH = os.environ.get("SNAPSHOT_PATH", "./data/movies_snapshot.parquet")


def validate():
    
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
