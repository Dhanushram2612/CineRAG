import json

from . import config, vectorstore, omdb_client, llm_client, people
from .memory import ConversationMemory

SYSTEM_PROMPT = (
    "You are a knowledgeable film recommender. You will be given a numbered "
    "list of candidate movies, each with its title, genres, rating, "
    "director/cast (when known), and a short overview. One candidate may "
    "be marked TOP PICK — the highest rating among reliably-voted "
    "candidates (vote count over 50).\n\n"
    "Choose the best 3-5 candidate NUMBERS for the user's request. You "
    "MUST choose only from the numbered candidates given — never invent, "
    "substitute, or reference any movie that is not one of these numbered "
    "candidates, even if you know of a more famous or better-fitting "
    "film. If nothing in the list fits well, return an empty selection "
    "and explain why in the note field, rather than forcing a match.\n\n"
    "Respond with ONLY valid JSON — no markdown code fences, no "
    "commentary before or after, nothing but the JSON object itself. Use "
    "exactly this shape:\n"
    '{\n'
    '  "recommended": [n, n, n],\n'
    '  "reasons": {"n": "1-2 sentence reason citing specific details from '
    'that candidate\'s overview, director, or cast — not just its genre '
    'label", ...},\n'
    '  "note": "optional short remark, e.g. why nothing fit well, or '
    'empty string otherwise"\n'
    '}\n'
    "Every key in \"reasons\" must be one of the numbers in \"recommended\", "
    "as a string. Do not output anything outside this JSON object."
)


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
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
            f"[{i}] {c['title']} | genres: {c['genres']} | rating: {rating}/10{credit_line}{top_pick_line}\n"
            f"  overview: {c.get('live_overview') or c['overview_snippet']}"
        )
    return "\n".join(lines)


def _mark_top_pick(candidates: list[dict]) -> None:
    if not candidates:
        return
    best = max(candidates, key=lambda c: c.get("live_vote_average") or c.get("vote_average") or 0)
    best["is_top_pick"] = True


_MUSIC_HINTS = {"music", "score", "soundtrack", "composer", "composed"}
_WRITING_HINTS = {"screenplay", "written", "writer", "story", "wrote"}

_HINT_BONUS = 0.03


def _resolve_person(search_query: str) -> tuple[str | None, str | None]:
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
    search_query = memory.condense_query(user_message)

    person_field, resolved_person = _resolve_person(user_message)
    if not resolved_person:
        person_field, resolved_person = _resolve_person(search_query)

    if resolved_person:
        candidates = _search_for_person(
            search_query, person_field, resolved_person, language=language,
            min_vote_count=config.MIN_VOTE_COUNT, min_vote_average=config.MIN_VOTE_AVERAGE,
        )
        if not candidates:
            candidates = _search_for_person(
                search_query, person_field, resolved_person,
                min_vote_count=config.MIN_VOTE_COUNT,
            )
        if not candidates:
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
        raw_reply = llm_client.chat(messages)
    except llm_client.LLMError as e:
        reply = f"Sorry, I hit an error generating a recommendation: {e}"
        selected_candidates = candidates
    else:
        reply, selected_candidates = _build_grounded_reply(raw_reply, candidates)

    memory.add_user(user_message)
    memory.add_assistant(reply)
    return {"reply": reply, "candidates": selected_candidates}


def _parse_llm_json(raw_reply: str) -> dict | None:
    text = raw_reply.strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict) or "recommended" not in data:
        return None
    return data


def _build_grounded_reply(raw_reply: str, candidates: list[dict]) -> tuple[str, list[dict]]:
    data = _parse_llm_json(raw_reply)

    if data is None:
        return _fallback_reply(candidates, reason="formatting"), candidates

    recommended_raw = data.get("recommended") or []
    reasons_raw = data.get("reasons") or {}
    note = (data.get("note") or "").strip()

    selected: list[tuple[int, dict]] = []
    seen_indices: set[int] = set()
    for n in recommended_raw:
        try:
            num = int(n)
        except (TypeError, ValueError):
            continue
        idx = num - 1
        if 0 <= idx < len(candidates) and idx not in seen_indices:
            seen_indices.add(idx)
            selected.append((num, candidates[idx]))

    if not selected:
        if note:
            return note, []
        return _fallback_reply(candidates, reason="no_selection"), candidates

    lines = []
    for i, (num, c) in enumerate(selected, start=1):
        rating = c.get("live_vote_average") or c.get("vote_average")
        badge = " 🏆 TOP PICK" if c.get("is_top_pick") else ""
        reason = reasons_raw.get(str(num), "").strip()
        entry = f"**{i}. {c['title']}**{badge}  \n⭐ {rating}/10"
        if reason:
            entry += f"  \n{reason}"
        lines.append(entry)

    top_pick = next((c for _, c in selected if c.get("is_top_pick")), None)
    if top_pick:
        rating = top_pick.get("live_vote_average") or top_pick.get("vote_average")
        lines.append(
            f"\nIf you only have time for one, start with **{top_pick['title']}** "
            f"({rating}/10) — the highest-rated pick among these."
        )

    if note:
        lines.append(f"\n{note}")

    return "\n\n".join(lines), [c for _, c in selected]


def _fallback_reply(candidates: list[dict], reason: str) -> str:
    if not candidates:
        return "I couldn't find matching movies in the catalog for that."

    prefix = (
        "Here's what I found (the recommendation engine had trouble "
        "formatting its response, so here are the raw matches):"
        if reason == "formatting" else
        "Here's what I found:"
    )
    lines = [prefix]
    for c in candidates:
        rating = c.get("live_vote_average") or c.get("vote_average")
        badge = " 🏆 TOP PICK" if c.get("is_top_pick") else ""
        lines.append(f"- **{c['title']}**{badge} — ⭐ {rating}/10")
    return "\n".join(lines)
