import requests
from . import config


class LLMError(Exception):
    pass


def _chat_ollama(messages: list[dict]) -> str:
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={"model": config.OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.RequestException as e:
        raise LLMError(
            f"Couldn't reach Ollama at {config.OLLAMA_HOST}. "
            f"Is `ollama serve` running and is the model pulled "
            f"(`ollama pull {config.OLLAMA_MODEL}`)? Details: {e}"
        ) from e
    except (KeyError, ValueError) as e:
        raise LLMError(f"Unexpected response from Ollama: {e}") from e


def _chat_groq(messages: list[dict]) -> str:
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            json={"model": config.GROQ_MODEL, "messages": messages, "temperature": 0.4},
            timeout=30,
        )
        if resp.status_code == 429:
            raise LLMError("Groq rate limit hit — please try again in a moment.")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        raise LLMError(f"Groq request failed: {e}") from e
    except (KeyError, IndexError, ValueError) as e:
        raise LLMError(f"Unexpected response from Groq: {e}") from e


def chat(messages: list[dict]) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": "..."}]"""
    if config.LLM_BACKEND == "ollama":
        return _chat_ollama(messages)
    if config.LLM_BACKEND == "groq":
        return _chat_groq(messages)
    raise LLMError(f"Unknown LLM_BACKEND '{config.LLM_BACKEND}' — use 'ollama' or 'groq'.")
