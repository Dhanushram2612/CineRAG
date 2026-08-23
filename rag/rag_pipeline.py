"""
Ties everything together for one chat turn.
"""
from . import config, vectorstore, omdb_client, llm_client, people
from .memory import ConversationMemory

SYSTEM_PROMPT = (
    "You are a knowledgeable film recommender. You will be given a list of "
    "candidate movies, each with its title, genres, rating, director/cast "
    "(when known), and a short overview. One candidate is marked TOP PICK — "
    "it has the highest rating among reliably-voted candidates (vote count "
    "over 50). Recommend the best 3-5 matches for the user's request, and "
    "for each one explain WHY it fits using details from its overview, "
    "director, or cast — not just its genre label. When you present the "
    "list, explicitly call out the TOP PICK as the one to watch first if "
    "they only have time for one, and say why it's rated highest among the "
    "options. Only recommend movies from the candidate list. If nothing "
    "fits well, say so honestly instead of forcing a match."
)


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for c in candidates:
        # `or`, not `.get(key, default)` — OMDb sets live_vote_average to
        # None (not missing) when it has no rating, and .get()'s default
        # only applies to a genuinely MISSING key.
        rating = c.get("live_vote_average") or c.get("vote_average")
        credit_bits = []
        if c.get("director"):
            credit_bits.append(f"director: {c['director']}")
        if c.get("composer"):
            credit_bits.append(f"music: {c['composer']}")
        if c.get("writer"):
            credit_bits.append(f"writer: {c['writer']}")
        if c.get("cast"):
            credit_bits.append(f"cast: {c['cast']}")
        credit_line = f" | {' | '.join(credit_bits)}" if credit_bits else ""
        top_pick_line = " | *** TOP PICK ***" if c.get("is_top_pick") else ""
        lines.append(
            f"- {c['title']} | genres: {c['genres']} | rating: {rating}/10{credit_line}{top_pick_line}\n"
            f"  overview: {c.get('live_overview') or c['overview_snippet']}"
        )
    return "\n".join(lines)


def _mark_top_pick(candidates: list[dict]) -> None:
    """Marks the single highest-rated candidate as the "top pick" — a
    concrete answer to "just tell me which one to watch," rather than
    leaving the user to weigh 3-5 equally-presented options themselves.
    All candidates already passed MIN_VOTE_COUNT (>50 votes) and
    MIN_VOTE_AVERAGE at retrieval time, so "highest rated among these" is
    already "highest rated among reliably-voted, decent options" — no
    extra filtering needed here, just picking the max."""
    if not candidates:
        return
    best = max(candidates, key=lambda c: c.get("live_vote_average") or c.get("vote_average") or 0)
    best["is_top_pick"] = True


# Keyword hints act as a tie-breaker bonus, not a hard priority order — see
# _resolve_person below. Kept from before: without these, "hans zimmer
# music" could favor an unrelated director/actor match over the composer
# field even at equal confidence.
_MUSIC_HINTS = {"music", "score", "soundtrack", "composer", "composed"}
_WRITING_HINTS = {"screenplay", "written", "writer", "story", "wrote"}

# Small nudge added to a field's score when the query contains a matching
# keyword hint — enough to break a genuine near-tie in that field's favor,
# but not enough to let a weak/coincidental match in the hinted field beat
# a much stronger match in a different field. (0.78 cutoff means anything
# that resolves at all is already >=0.78; a 0.03 nudge only matters when
# two fields are within a few hundredths of each other.)
_HINT_BONUS = 0.03


def _resolve_person(search_query: str) -> tuple[str | None, str | None]:
    """Tries ALL FOUR person-type resolvers (director/actor/composer/
    writer) and returns whichever produced the HIGHEST-CONFIDENCE match,
    rather than the first resolver to clear the cutoff.

    This replaces an earlier version that stopped at the first resolver in
    a fixed priority order (director, then actor, then composer, then
    writer). That approach had a real bug: at real catalog scale (thousands
    of names), a query can produce a weak, coincidental fuzzy match in an
    EARLIER-tried field (e.g. "jake" fuzzy-matching an unrelated director's
    surname at ~0.80) that silently wins over a much stronger, CORRECT
    match in a later-tried field (e.g. "gyllenhaal" matching the intended
    actor at 1.0) — purely because director was checked before actor, not
    because it was actually the better match. Comparing scores across all
    four fields and keeping the best one fixes that class of bug entirely,
    regardless of which field happens to be checked first.

    Keyword hints (_MUSIC_HINTS/_WRITING_HINTS) still apply, but only as a
    small tie-breaking bonus on top of the real match score — not as a
    reason to skip checking the other fields."""
    words = {w.strip(".,!?") for w in search_query.lower().split()}

    resolvers: list[tuple[str, callable]] = [
        ("director", people.resolve_director_scored),
        ("actor", people.resolve_actor_scored),
        ("composer", people.resolve_composer_scored),
        ("writer", people.resolve_writer_scored),
    ]

    best_field, best_name, best_score = None, None, 0.0
    for field_name, resolver in resolvers:
        name, score = resolver(search_query)
        if not name:
            continue
        if field_name == "composer" and (words & _MUSIC_HINTS):
            score += _HINT_BONUS
        if field_name == "writer" and (words & _WRITING_HINTS):
            score += _HINT_BONUS
        if score > best_score:
            best_field, best_name, best_score = field_name, name, score

    return best_field, best_name


def _search_for_person(search_query: str, field: str, name: str, **extra_filters) -> list[dict]:
    """Dispatches to the right vectorstore.search() kwarg for whichever
    person-field resolved — director/composer/writer use an exact metadata
    match, actor uses the document-content substring match (see
    vectorstore.search's docstring for why cast is different)."""
    kwargs = dict(query_text=search_query, n_results=config.TOP_K, **extra_filters)
    if field == "director":
        kwargs["director"] = name
    elif field == "actor":
        kwargs["cast_contains"] = name
    elif field == "composer":
        kwargs["composer"] = name
    elif field == "writer":
        kwargs["writer"] = name
    return vectorstore.search(**kwargs)


def answer(user_message: str, memory: ConversationMemory,
           language: str | None = None) -> dict:
    """Returns {"reply": str, "candidates": list[dict]} so the UI can render
    both the LLM's text and structured movie cards (with posters)."""

    search_query = memory.condense_query(user_message)

    # Semantic search alone is unreliable for exact-name recall — a query
    # like "David Fincher's best," "jake gyllenhaal movies," or "hans
    # zimmer music" can miss the actual films if they don't rank in the
    # top-N purely on thematic similarity. Try resolving a specific person
    # (director/actor/composer/writer) against real catalog names first.
    #
    # Resolve against the RAW user message first, not the condensed query —
    # condense_query() is LLM-based and built for multi-turn coreference
    # ("his movies" -> "Jake Gyllenhaal's movies"), but on plain single-turn
    # queries it can also paraphrase away the exact name it was given
    # (e.g. rewriting into a generic "recommend top rated films"). Falling
    # back to the condensed query second still covers the coreference case
    # where the name only exists there.
    person_field, resolved_person = _resolve_person(user_message)
    if not resolved_person:
        person_field, resolved_person = _resolve_person(search_query)

    if resolved_person:
        # Staged fallback that NEVER drops the person constraint — only
        # relaxes the OTHER filters around it. Silently dropping the person
        # filter (an earlier version of this code did exactly that) means a
        # "Jake Gyllenhaal movies" query could quietly return movies with no
        # connection to him at all once filters got too strict — confidently
        # wrong is worse than an honest "not enough results."
        candidates = _search_for_person(
            search_query, person_field, resolved_person, language=language,
            min_vote_count=config.MIN_VOTE_COUNT, min_vote_average=config.MIN_VOTE_AVERAGE,
        )
        if not candidates:
            # Drop language + rating floor, keep vote-count sanity check.
            candidates = _search_for_person(
                search_query, person_field, resolved_person,
                min_vote_count=config.MIN_VOTE_COUNT,
            )
        if not candidates:
            # Last resort: person constraint only, no quality/vote filters
            # at all — covers a real person with very few, lightly-voted
            # films in this dataset.
            candidates = _search_for_person(search_query, person_field, resolved_person)
        if not candidates:
            reply = (f"I couldn't find any movies involving {resolved_person} in the "
                      "current catalog at all. Their films may just not be in this "
                      "dataset — this isn't the same as them having no good movies.")
            memory.add_user(user_message)
            memory.add_assistant(reply)
            return {"reply": reply, "candidates": []}
    else:
        candidates = vectorstore.search(
            query_text=search_query, n_results=config.TOP_K, language=language,
            min_vote_count=config.MIN_VOTE_COUNT, min_vote_average=config.MIN_VOTE_AVERAGE,
        )
        if not candidates:
            reply = ("I couldn't find matching movies in the catalog for that. "
                      "Try rephrasing, or widen the language filter.")
            memory.add_user(user_message)
            memory.add_assistant(reply)
            return {"reply": reply, "candidates": []}

    # Live-enrich only the retrieved top-k, not the whole catalog.
    enrichment_data = omdb_client.enrich_many(candidates)
    for c in candidates:
        detail = enrichment_data.get(c["id"])
        if detail:
            c["live_vote_average"] = detail.get("vote_average")
            c["live_overview"] = detail.get("overview")
            c["poster_url"] = detail.get("poster_url")

    _mark_top_pick(candidates)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += memory.history_messages()
    messages.append({
        "role": "user",
        "content": f"User request: {user_message}\n\n"
                    f"Candidate movies:\n{_format_candidates(candidates)}",
    })

    try:
        reply = llm_client.chat(messages)
    except llm_client.LLMError as e:
        reply = f"Sorry, I hit an error generating a recommendation: {e}"

    memory.add_user(user_message)
    memory.add_assistant(reply)
    return {"reply": reply, "candidates": candidates}