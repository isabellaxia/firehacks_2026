"""Featherless client.

One key, many model families. The whole architecture depends on being able to call
Qwen, Llama and Mistral through the same endpoint without deploying anything.

Two guards that matter during a live demo:
  * a semaphore sized to your plan's concurrent-request cap, so a fan-out doesn't 429
  * an in-memory cache, so the second run of your demo is instant
"""
import asyncio
import hashlib
import os

from openai import AsyncOpenAI

BASE_URL = os.environ.get("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")
API_KEY = os.environ.get("FEATHERLESS_API_KEY", "")

# Featherless plans cap simultaneous requests. Set this to your cap minus one.
CONCURRENCY = int(os.environ.get("FEATHERLESS_CONCURRENCY", "2"))

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY or "missing-key", timeout=90.0)
_sem = asyncio.Semaphore(CONCURRENCY)
_cache: dict[str, str] = {}


class LLMError(RuntimeError):
    pass


async def complete(model: str, system: str, user: str, max_tokens: int = 500,
                   temperature: float = 0.2) -> str:
    key = hashlib.sha256(f"{model}|{system}|{user}|{max_tokens}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    async with _sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                out = (r.choices[0].message.content or "").strip()
                _cache[key] = out
                return out
            except Exception as e:
                if attempt == 2:
                    raise LLMError(f"{model}: {e}") from e
                await asyncio.sleep(1.5 * (attempt + 1))
    return ""


async def list_models(limit: int = 0) -> list[str]:
    """GET /v1/models. Used at startup to confirm the configured models exist."""
    r = await client.models.list()
    ids = [m.id for m in r.data]
    return ids[:limit] if limit else ids

