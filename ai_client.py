"""
ATHR AI client (M7).

A thin, provider-agnostic wrapper around one HTTP call to an LLM API.
The model only understands the question and writes the answer; Mars data
never comes from here (it comes from ATHR's own adapters).

Configuration is read from environment variables / .env. Nothing is
hard-coded and the API key never leaves the backend.

  ATHR_AI_PROVIDER   anthropic | openai   (openai = any OpenAI-compatible API:
                                           OpenAI, Gemini compat, Groq, OpenRouter, Ollama...)
  ATHR_AI_API_KEY    secret key
  ATHR_AI_MODEL      model id, exactly as the provider names it
  ATHR_AI_BASE_URL   optional; override the provider's default URL
"""

import logging
import os

import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional; real env vars still work
    pass

log = logging.getLogger("athr.ai")

DEFAULT_BASE_URLS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
}
ANTHROPIC_VERSION = "2023-06-01"
MAX_OUTPUT_TOKENS = 700
TIMEOUT = httpx.Timeout(45.0, connect=10.0)


class AIConfigError(Exception):
    """The AI provider is not (correctly) configured."""


class AIServiceError(Exception):
    """The AI provider failed or was unreachable. Message is safe to show."""


def get_config() -> dict:
    provider = os.getenv("ATHR_AI_PROVIDER", "").strip().lower()
    api_key = os.getenv("ATHR_AI_API_KEY", "").strip()
    model = os.getenv("ATHR_AI_MODEL", "").strip()

    if provider not in DEFAULT_BASE_URLS:
        raise AIConfigError("ATHR_AI_PROVIDER must be 'anthropic' or 'openai'.")
    if not api_key:
        raise AIConfigError("ATHR_AI_API_KEY is not set.")
    if not model:
        raise AIConfigError("ATHR_AI_MODEL is not set.")

    base_url = os.getenv("ATHR_AI_BASE_URL", "").strip() or DEFAULT_BASE_URLS[provider]
    return {
        "provider": provider,
        "api_key": api_key,
        "model": model,
        "base_url": base_url.rstrip("/"),
    }


def is_configured() -> bool:
    try:
        get_config()
        return True
    except AIConfigError:
        return False


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    # Upstream body goes to the server log only (it may be long or detailed).
    log.warning("AI provider HTTP %s: %s", response.status_code, response.text[:500])
    code = response.status_code
    if code in (401, 403):
        raise AIServiceError("The AI provider rejected the credentials (check the API key).")
    if code == 404:
        raise AIServiceError("The AI provider could not find the model or endpoint (check model and base URL).")
    if code == 429:
        raise AIServiceError("The AI provider rate limit or quota was reached. Try again shortly.")
    if code >= 500:
        raise AIServiceError("The AI provider is temporarily unavailable.")
    raise AIServiceError(f"The AI provider rejected the request (HTTP {code}).")


def generate(system: str, messages: list[dict]) -> str:
    """Send `messages` ([{role, content}], user first) and return the reply text."""
    cfg = get_config()

    if cfg["provider"] == "anthropic":
        url = f"{cfg['base_url']}/v1/messages"
        headers = {
            "x-api-key": cfg["api_key"],
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        body = {
            "model": cfg["model"],
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": system,
            "messages": messages,
        }
    else:
        url = f"{cfg['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "content-type": "application/json",
        }
        # Some newer OpenAI models want "max_completion_tokens" instead.
        token_param = os.getenv("ATHR_AI_TOKEN_PARAM", "max_tokens")
        body = {
            "model": cfg["model"],
            token_param: MAX_OUTPUT_TOKENS,
            "messages": [{"role": "system", "content": system}, *messages],
        }

    try:
        response = httpx.post(url, headers=headers, json=body, timeout=TIMEOUT)
    except httpx.TimeoutException:
        raise AIServiceError("The AI provider timed out.") from None
    except httpx.HTTPError as exc:
        log.warning("AI provider unreachable: %s", exc)
        raise AIServiceError("The AI provider could not be reached.") from None

    _raise_for_status(response)

    try:
        payload = response.json()
        if cfg["provider"] == "anthropic":
            text = "".join(
                block.get("text", "")
                for block in payload.get("content", [])
                if block.get("type") == "text"
            )
        else:
            text = payload["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise AIServiceError("The AI provider returned an unreadable response.") from None

    text = text.strip()
    if not text:
        raise AIServiceError("The AI provider returned an empty answer.")
    return text