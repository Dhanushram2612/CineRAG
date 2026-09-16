import difflib

from . import vectorstore

_DIRECTOR_NAMES_CACHE: list[str] | None = None
_ACTOR_NAMES_CACHE: list[str] | None = None
_COMPOSER_NAMES_CACHE: list[str] | None = None
_WRITER_NAMES_CACHE: list[str] | None = None


_STOPWORDS = {
    # original set
    "best", "movie", "movies", "film", "films", "the", "a", "an", "please",
    "show", "me", "recommend", "some", "good", "great", "top", "directed",
    "by", "director", "something", "shorter", "longer", "less", "more",
    "starring", "acted", "actor", "actress", "with", "in", "music",
    "score", "soundtrack", "composer", "composed", "screenplay", "written",
    "writer", "story", "wrote","to", "from", "and", "or", "but", "nor", "as", 
    "at", "on", "of", "for","into", "onto", "throughout", "through", "across", 
    "over", "under","about", "than", "then", "if", "so", "that", "this", "these", "those",
    "it", "its", "i", "you", "your", "we", "our", "us", "he", "she",
    "they", "them", "his", "her", "their", "is", "was", "are", "were",
    "be", "been", "being", "stays", "stay", "staying", "has", "have",
    "had", "do", "does", "did", "not", "no", "yes",
    "want", "need", "give", "suggest", "suggestion", "suggestions",
    "find", "search", "look", "looking", "for", "genre", "genres", "type",
    "kind", "sort", "sorts", "engaging", "tense", "tension", "sustained",
    "start", "end", "beginning", "throughout","worth", "watch", "forrest",
}


def _load_director_names() -> list[str]:
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
    col = vectorstore.get_collection()
    all_meta = col.get(include=["metadatas"])["metadatas"]
    names = {m.get(field, "") for m in all_meta if m.get(field)}
    return sorted(names)


def _load_actor_names() -> list[str]:
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
    name, _ = _fuzzy_resolve(text, _load_director_names(), cutoff)
    return name


def resolve_actor(text: str, cutoff: float = 0.78) -> str | None:
    name, _ = _fuzzy_resolve(text, _load_actor_names(), cutoff)
    return name


def resolve_composer(text: str, cutoff: float = 0.78) -> str | None:
    name, _ = _fuzzy_resolve(text, _load_composer_names(), cutoff)
    return name


def resolve_writer(text: str, cutoff: float = 0.78) -> str | None:
    name, _ = _fuzzy_resolve(text, _load_writer_names(), cutoff)
    return name


def resolve_director_scored(text: str, cutoff: float = 0.78) -> tuple[str | None, float]:
    return _fuzzy_resolve(text, _load_director_names(), cutoff)


def resolve_actor_scored(text: str, cutoff: float = 0.78) -> tuple[str | None, float]:
    return _fuzzy_resolve(text, _load_actor_names(), cutoff)


def resolve_composer_scored(text: str, cutoff: float = 0.78) -> tuple[str | None, float]:
    return _fuzzy_resolve(text, _load_composer_names(), cutoff)


def resolve_writer_scored(text: str, cutoff: float = 0.78) -> tuple[str | None, float]:
    return _fuzzy_resolve(text, _load_writer_names(), cutoff)


def looks_like_director_query(text: str) -> bool:
    return resolve_director(text) is not None
