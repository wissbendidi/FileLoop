"""
Optional Ollama client for local LLM answers.

When Ollama is running (e.g. http://localhost:11434), the Chat page can
generate natural-language answers from retrieved file snippets. No data
is sent outside the machine.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Optional, List

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
TIMEOUT_SEC = 30


def check_ollama_available(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Return True if Ollama is reachable."""
    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception as e:
        logger.debug("Ollama not available: %s", e)
        return False


def list_models(base_url: str = DEFAULT_BASE_URL) -> List[str]:
    """Return list of available model names. Empty if Ollama unavailable."""
    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        models = data.get("models", [])
        return [m.get("name", "").split(":")[0] for m in models if m.get("name")]
    except Exception as e:
        logger.debug("Could not list Ollama models: %s", e)
        return []


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    system: Optional[str] = None,
    timeout: int = TIMEOUT_SEC,
) -> Optional[str]:
    """
    Send prompt to Ollama and return the generated response text, or None on error.
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        body["system"] = system

    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode())
        return out.get("response", "").strip() or None
    except Exception as e:
        logger.warning("Ollama generate failed: %s", e)
        return None
