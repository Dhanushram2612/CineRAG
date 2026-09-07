"""
Detects when a user's query is really asking about a specific director or
actor, and resolves fuzzy/misspelled/lowercase names against the actual
people present in the catalog — so rag_pipeline.py can route those queries
through a targeted filter (metadata match for director, document-content
match for cast) instead of relying on pure semantic similarity, which is
unreliable for exact-name recall.

Example: "david finchers best" -> word n-grams fuzzy-matched (case-
insensitively) against catalog directors -> resolves to "David Fincher" ->
used as an exact metadata filter, combined with semantic ranking within
his films.

Example: "jake gyllenhaal movies" -> resolves to "Jake Gyllenhaal" against
the catalog's cast names -> used as a document-content substring filter
(cast is a joined "Actor1, Actor2, ..." string per movie, not a single
metadata value, so it can't use an exact metadata match the way director
does — see resolve_actor for why this uses where_document instead).

Note: detection and resolution are combined into one step here, not two.
An earlier version tried to first detect "is this director-intent?" via a
regex requiring capitalized words, then only resolve if so — but real users
type lowercase ("david finchers best", not "David Fincher's best"), so a
capitalization-dependent heuristic silently failed on exactly the case it
was meant to catch. Attempting fuzzy resolution directly against every
query is more robust and doesn't depend on how the user capitalized anything.
"""
import difflib

from . import vectorstore

_DIRECTOR_NAMES_CACHE: list[str] | None = None
_ACTOR_NAMES_CACHE: list[str] | None = None
_COMPOSER_NAMES_CACHE: list[str] | None = None
_WRITER_NAMES_CACHE: list[str] | None = None

# Words that are unlikely to ever be part of a person's name, filtered out
# of candidate n-grams to reduce noise/false-positive fuzzy matches.
#
# This list matters more than it looks like it should: a short, common
# English word can be an EXACT match (score 1.0) against a real surname —
# e.g. "to" in "tense from start to end" is a perfect match for director
# Johnnie To. That's not a fuzzy near-miss the cutoff can catch; it's a
# genuine, deterministic collision. The only real defense is keeping common
# function words (prepositions, conjunctions, pronouns, auxiliary verbs)
# out of the n-gram candidate pool entirely, so they never get compared
# against the name lists at all. Deliberately NOT filtering by word length
# instead (e.g. "skip anything under 3 letters") — that would also break
# genuinely short real surnames like "Ng", "Wu", or "To" itself when a user
# actually means to search for that person.
_STOPWORDS = {
    # original set
    "best", "movie", "movies", "film", "films", "the", "a", "an", "please",
    "show", "me", "recommend", "some", "good", "great", "top", "directed",
    "by", "director", "something", "shorter", "longer", "less", "more",
    "starring", "acted", "actor", "actress", "with", "in", "music",
    "score", "soundtrack", "composer", "composed", "screenplay", "written",
    "writer", "story", "wrote",
    # prepositions / conjunctions — the main gap that let "to" (-> Johnnie
    # To) through
    "to", "from", "and", "or", "but", "nor", "as", "at", "on", "of", "for",
    "into", "onto", "throughout", "through", "across", "over", "under",
    "about", "than", "then", "if", "so", "that", "this", "these", "those",
    # pronouns / auxiliary & common verbs
    "it", "its", "i", "you", "your", "we", "our", "us", "he", "she",
    "they", "them", "his", "her", "their", "is", "was", "are", "were",
    "be", "been", "being", "stays", "stay", "staying", "has", "have",
    "had", "do", "does", "did", "not", "no", "yes",
    # generic request/quality words that show up constantly in movie
    # prompts but are never part of a real name
    "want", "need", "give", "suggest", "suggestion", "suggestions",
    "find", "search", "look", "looking", "for", "genre", "genres", "type",
    "kind", "sort", "sorts", "engaging", "tense", "tension", "sustained",
    "start", "end", "beginning", "throughout",
    # Words that happen to be real (usually short/common) surnames
    # somewhere in this catalog — "worth" collides with director David
    # Worth / composer Stan Worth, "forrest" collides with actor Frederic
    # Forrest. These are genuine 1.0-confidence exact matches, not fuzzy
    # noise, so a cutoff adjustment can't filter them out — keeping them
    # out of the n-gram pool is the only reliable fix. Expect to keep
    # adding entries here as new collisions surface; it's an inherent
    # cost of fuzzy-matching against a large real-world name list rather
    # than a bug that gets "solved" once.
    "worth", "watch", "forrest",
}


def _load_director_names() -> list[str]:
    """Pulls every distinct, non-empty director name out of the Chroma
    collection's metadata. Cached at module level since the catalog only
    changes when ingest.py is re-run (a new process start), not per-query."""
    global _DIRECTOR_NAMES_CACHE
    if _DIRECTOR_NAMES_CACHE is not None:
        return _DIRECTOR_NAMES_CACHE
    _DIRECTOR_NAMES_CACHE = _load_single_value_names("director")
    return _DIRECTOR_NAMES_CACHE


def _load_composer_names() -> list[str]:
    global _COMPOSER_NAMES_CACHE
    if _COMPOSER_NAMES_CACHE is not None:
        return _COMPOSER_NAMES_CACHE
    _COMPOSER_NAMES_CACHE = _load_single_value_names("composer")
    return _COMPOSER_NAMES_CACHE


def _load_writer_names() -> list[str]:
    global _WRITER_NAMES_CACHE
    if _WRITER_NAMES_CACHE is not None:
        return _WRITER_NAMES_CACHE
    _WRITER_NAMES_CACHE = _load_single_value_names("writer")
    return _WRITER_NAMES_CACHE


def _load_single_value_names(field: str) -> list[str]:
    """Shared loader for any metadata field that holds one name per movie
    (director, composer, writer) — unlike 'cast', which holds several names
    joined into one string and needs _load_actor_names' splitting logic."""
    col = vectorstore.get_collection()
    all_meta = col.get(include=["metadatas"])["metadatas"]
    names = {m.get(field, "") for m in all_meta if m.get(field)}
    return sorted(names)


def _load_actor_names() -> list[str]:
    """Same idea as _load_director_names, but the 'cast' metadata field is a
    single joined string per movie ("Actor1, Actor2, ..., Actor5") rather
    than one name — so this splits every movie's cast string apart into
    individual names before deduplicating."""
    global _ACTOR_NAMES_CACHE
    if _ACTOR_NAMES_CACHE is not None:
        return _ACTOR_NAMES_CACHE

    col = vectorstore.get_collection()
    all_meta = col.get(include=["metadatas"])["metadatas"]
    names: set[str] = set()
    for m in all_meta:
        cast_str = m.get("cast", "")
        if cast_str:
            names.update(n.strip() for n in cast_str.split(",") if n.strip())
    _ACTOR_NAMES_CACHE = sorted(names)
    return _ACTOR_NAMES_CACHE


def _candidate_ngrams(text: str) -> list[str]:
    """Generates 1-3 word sliding windows from the query, case-insensitively,
    skipping pure stopword tokens. Deliberately doesn't require
    capitalization — real user input is often lowercase."""
    words = [w.strip(".,!?'\"") for w in text.split()]
    words = [w for w in words if w]

    ngrams = []
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            window = words[i:i + size]
            if all(w.lower() in _STOPWORDS for w in window):
                continue
            ngrams.append(" ".join(window))
    return ngrams


def _fuzzy_resolve(text: str, known_names: list[str], cutoff: float) -> tuple[str | None, float]:
    """Shared matching core for both directors and actors: tries every
    n-gram against full names AND surnames alone (most people say "nolan
    movies," not "Christopher Nolan movies"), case-insensitively, tolerant
    of a missing apostrophe ("finchers" for "Fincher's").

    Returns (matched_name, score) instead of just the name — score is
    needed by _resolve_person (in rag_pipeline.py) to compare candidates
    ACROSS different person-types (director vs actor vs composer vs
    writer), not just within one. Without the score, a weak coincidental
    match in one field (e.g. "jake" fuzzy-matching a mostly-unrelated
    director's surname at ~0.80) can silently win over a much stronger,
    correct match in a different field (e.g. "gyllenhaal" matching an
    actor at 1.0) purely because that resolver happened to be tried first.
    At catalog scale (thousands of real names), these coincidental
    same-cutoff collisions are common enough that "first resolver to
    clear the cutoff wins" isn't reliable — highest score across all
    fields should win instead."""
    if not known_names:
        return None, 0.0

    full_lookup = {d.lower(): d for d in known_names}
    surname_lookup: dict[str, str] = {}
    for d in known_names:
        surname = d.split()[-1].lower() if d.split() else ""
        if surname and surname not in surname_lookup:
            surname_lookup[surname] = d

    best_match = None
    best_score = 0.0
    for ngram in _candidate_ngrams(text):
        cleaned = ngram.lower()
        if cleaned.endswith("s") and not cleaned.endswith("ss"):
            cleaned = cleaned[:-1]

        for lookup in (full_lookup, surname_lookup):
            matches = difflib.get_close_matches(cleaned, list(lookup.keys()), n=1, cutoff=cutoff)
            if matches:
                score = difflib.SequenceMatcher(None, cleaned, matches[0]).ratio()
                if score > best_score:
                    best_match, best_score = lookup[matches[0]], score

    return best_match, best_score


def resolve_director(text: str, cutoff: float = 0.78) -> str | None:
    """Returns the catalog's exact spelling of a director mentioned in the
    query, or None if nothing matched with reasonable confidence.

    cutoff=0.78 is a starting point, not a rigorously tuned value — if this
    produces false-positive matches in practice, raise it; if real director
    queries aren't matching due to typos, lower it slightly. This function
    doubles as the intent check: if nothing resolves, treat the query as not
    being about a specific director and fall back to plain semantic search."""
    name, _ = _fuzzy_resolve(text, _load_director_names(), cutoff)
    return name


def resolve_actor(text: str, cutoff: float = 0.78) -> str | None:
    """Same idea as resolve_director, but for cast members. Returns the
    catalog's exact spelling of the matched actor's name, intended to be
    used as a document-content substring filter (see vectorstore.search's
    cast_contains param) rather than a metadata equality filter, since the
    'cast' field holds multiple joined names per movie, not one."""
    name, _ = _fuzzy_resolve(text, _load_actor_names(), cutoff)
    return name


def resolve_composer(text: str, cutoff: float = 0.78) -> str | None:
    """Resolves a query like "hans zimmer music" or "zimmer score" against
    the catalog's composer names."""
    name, _ = _fuzzy_resolve(text, _load_composer_names(), cutoff)
    return name


def resolve_writer(text: str, cutoff: float = 0.78) -> str | None:
    """Resolves a query like "aaron sorkin screenplay" against the
    catalog's writer/screenwriter names."""
    name, _ = _fuzzy_resolve(text, _load_writer_names(), cutoff)
    return name


# --- Score-returning variants ---
# rag_pipeline._resolve_person needs the confidence score to compare
# matches ACROSS person-types (see _fuzzy_resolve's docstring) — the
# name-only resolve_*() functions above stay as the public, simple-contract
# API for any other/future caller that just wants a yes/no name.

def resolve_director_scored(text: str, cutoff: float = 0.78) -> tuple[str | None, float]:
    return _fuzzy_resolve(text, _load_director_names(), cutoff)


def resolve_actor_scored(text: str, cutoff: float = 0.78) -> tuple[str | None, float]:
    return _fuzzy_resolve(text, _load_actor_names(), cutoff)


def resolve_composer_scored(text: str, cutoff: float = 0.78) -> tuple[str | None, float]:
    return _fuzzy_resolve(text, _load_composer_names(), cutoff)


def resolve_writer_scored(text: str, cutoff: float = 0.78) -> tuple[str | None, float]:
    return _fuzzy_resolve(text, _load_writer_names(), cutoff)


def looks_like_director_query(text: str) -> bool:
    """Kept as a separate public function for callers that want a cheap
    yes/no without needing the resolved name — but note it does the same
    fuzzy-matching work internally, since a capitalization/keyword-based
    shortcut proved unreliable (see module docstring)."""
    return resolve_director(text) is not None