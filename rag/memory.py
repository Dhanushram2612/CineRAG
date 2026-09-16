from . import llm_client

MAX_TURNS_KEPT = 6  # trailing turns kept in context, to bound prompt size

CONDENSE_SYSTEM_PROMPT = (
    "You rewrite a user's latest message into a single, standalone movie "
    "search query, using the conversation so far for context. Output ONLY "
    "the rewritten query, nothing else. If the latest message is already "
    "standalone, return it unchanged."
)


class ConversationMemory:
    def __init__(self):
        self.turns: list[dict] = []  # [{"role": "user"/"assistant", "content": str}]

    def add_user(self, text: str):
        self.turns.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.turns.append({"role": "assistant", "content": text})
        self.turns = self.turns[-MAX_TURNS_KEPT:]

    def history_messages(self) -> list[dict]:
        return list(self.turns[-MAX_TURNS_KEPT:])

    def condense_query(self, latest_message: str) -> str:
        """Turn a context-dependent follow-up into a standalone search query."""
        if not self.turns:
            return latest_message

        history_text = "\n".join(f"{t['role']}: {t['content']}" for t in self.turns[-4:])
        messages = [
            {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation so far:\n{history_text}\n\n"
                                         f"Latest message: {latest_message}"},
        ]
        try:
            return llm_client.chat(messages).strip().strip('"')
        except llm_client.LLMError:
            # If condensation fails, fall back to the raw message rather than
            # blocking the whole turn on a non-critical step.
            return latest_message
