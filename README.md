# CineRAG

A conversational movie-recommendation RAG system: ChromaDB for retrieval,
Ollama for local development, Groq for cloud deployment, OMDb for live data.

## Architecture

```
Kaggle CSV (ingest.py, offline) --> parquet snapshot --> ChromaDB (cosine)
                                                              |
user message -> condense w/ history -> retrieve top-k -------|
                                                              v
                                            live enrich via OMDb (top-k only)
                                                              |
                                                              v
                                     LLM (Ollama local / Groq cloud)
                                                              |
                                                              v
                                              Streamlit chat UI
```

## Why OMDb

TMDB's API/website is blocked on some networks (bundled into anti-piracy
blocklists on parts of the Indian network, unrelated to TMDB itself), which
breaks anything depending on it for local development. So this project
avoids the dependency entirely:

- **Catalog building** (`ingest.py`) uses the static Kaggle
  "tmdb-movie-metadata" CSV export — a one-time file download, not a live
  API call. Works everywhere, every time.
- **Live enrichment** (`omdb_client.py`) uses OMDb — a different
  provider/domain, free, and only needs title lookup, which is exactly
  what's needed to refresh an already-retrieved shortlist (~5-8
  movies/query, well under its 1,000/day free limit).

## Other key design decisions and why

- **Overview text reaches the LLM.** Retrieval used the overview, but the
  LLM only ever saw a title and genre label, so its reasoning was generic.
  Now the full candidate context (title, genres, rating, overview) goes into
  the prompt.
- **Cosine similarity, not raw L2.** Chroma's `hnsw:space: cosine` avoids the
  magnitude bias that unnormalized L2 distance introduces on sentence
  embeddings.
- **Snapshot + live enrichment, not full live re-embedding.** Re-fetching and
  re-embedding the whole catalog on every cold start would be slow and
  wasteful on a free host. `ingest.py` builds a cached parquet snapshot
  offline; the app loads that instantly, then enriches live only the handful
  of movies actually retrieved.
- **Multi-turn chat with query condensation.** Follow-ups like "something
  shorter" are rewritten into a standalone search query using the
  conversation history before being embedded — a bare embedding model has no
  memory of its own.
- **No hardcoded secrets.** Everything comes from environment variables /
  host secrets. See `.env.example`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OMDB_API_KEY, GROQ_API_KEY
```

Get a free OMDb key: https://www.omdbapi.com/apikey.aspx

### Build the catalog

```bash
# Download tmdb_5000_movies.csv from Kaggle ("tmdb-movie-metadata" dataset)
python -m rag.ingest --movies tmdb_5000_movies.csv
```

### Local development (Ollama)

```bash
ollama serve
ollama pull llama3.1:8b
# LLM_BACKEND=ollama in .env
streamlit run app.py
```

### Deployment (Groq, e.g. HF Spaces / Streamlit Cloud)

Set `LLM_BACKEND=groq` and `GROQ_API_KEY` in your host's secrets manager (not
in the repo). Commit the `data/movies_snapshot.parquet` snapshot (not the raw
chroma DB) so the deployed app has data to load on startup — it rebuilds the
Chroma index from the snapshot on first launch.

## Refreshing the catalog

```bash
python -m rag.ingest --movies tmdb_5000_movies.csv
```

Re-download the CSV from Kaggle periodically and re-run to pick up dataset
updates. Run this offline (cron, GitHub Action) — not on every app request.
