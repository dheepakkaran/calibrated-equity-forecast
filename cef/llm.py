"""Language-model providers for the narration layers.

Two providers, in a deliberate order. Gemini's free tier is tried first because
the simple view is a long generation run once per symbol per session — across
52 symbols that is real money on a paid API and nothing on a free one. OpenAI
is the fallback and the default for the short technical narration.

Both are reached over plain HTTP with ``requests`` rather than through vendor
SDKs. The payloads here are small and stable, and two SDKs would mean two
dependency-version treadmills for what amounts to one POST each.

Neither provider is ever asked to compute anything. Every number in every
prompt was produced by the pipeline, and the response is checked against those
numbers before it is shown.
"""
from __future__ import annotations

import json
import logging
import os

import requests

from cef.config import OPENAI_API_KEY

log = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
OPENAI_BASE = "https://api.openai.com/v1"
OPENAI_MODEL = os.getenv("NARRATOR_MODEL", "gpt-5-mini")

TIMEOUT = 120


class NoProvider(RuntimeError):
    pass


def available() -> list[str]:
    out = []
    if GEMINI_API_KEY:
        out.append("gemini")
    if OPENAI_API_KEY:
        out.append("openai")
    return out


def gemini_models() -> list[str]:
    """Ask the API which models this key can actually use, rather than
    hard-coding a name that may have been retired."""
    if not GEMINI_API_KEY:
        return []
    r = requests.get(f"{GEMINI_BASE}/models", params={"key": GEMINI_API_KEY},
                     timeout=TIMEOUT)
    r.raise_for_status()
    return [m["name"].split("/")[-1] for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])]


def _gemini_json(system: str, user: str, schema: dict, model: str) -> tuple[dict, dict]:
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": 0.2,
        },
    }
    r = requests.post(f"{GEMINI_BASE}/models/{model}:generateContent",
                      params={"key": GEMINI_API_KEY}, json=body, timeout=TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"gemini {r.status_code}: {r.text[:300]}")
    data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    usage = data.get("usageMetadata", {})
    return json.loads(text), {
        "provider": "gemini", "model": model,
        "tokens_in": usage.get("promptTokenCount"),
        "tokens_out": usage.get("candidatesTokenCount"),
    }


def _openai_json(system: str, user: str, schema: dict, model: str) -> tuple[dict, dict]:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "simple_view", "strict": False, "schema": schema},
        },
    }
    r = requests.post(f"{OPENAI_BASE}/chat/completions",
                      headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                      json=body, timeout=TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"openai {r.status_code}: {r.text[:300]}")
    data = r.json()
    usage = data.get("usage", {})
    return json.loads(data["choices"][0]["message"]["content"]), {
        "provider": "openai", "model": model,
        "tokens_in": usage.get("prompt_tokens"),
        "tokens_out": usage.get("completion_tokens"),
    }


def generate_json(system: str, user: str, schema: dict,
                  prefer: str = "gemini") -> tuple[dict, dict]:
    """Structured generation, free provider first.

    Both providers support a response schema, so the shape of the output is
    enforced by the API rather than by parsing prose and hoping. A provider
    that errors is logged and skipped; only running out of providers raises.
    """
    order = (["gemini", "openai"] if prefer == "gemini" else ["openai", "gemini"])
    errors = []
    for name in order:
        try:
            if name == "gemini" and GEMINI_API_KEY:
                return _gemini_json(system, user, schema, GEMINI_MODEL)
            if name == "openai" and OPENAI_API_KEY:
                return _openai_json(system, user, schema, OPENAI_MODEL)
        except Exception as exc:                       # noqa: BLE001
            log.warning("%s failed: %s", name, str(exc)[:200])
            errors.append(f"{name}: {str(exc)[:160]}")
    raise NoProvider(
        "no language-model provider succeeded. Set GEMINI_API_KEY (free tier) "
        "or OPENAI_API_KEY in .env. " + (" | ".join(errors) if errors else ""))
