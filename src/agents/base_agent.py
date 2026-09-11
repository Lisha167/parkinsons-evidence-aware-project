"""
base_agent.py
---------------------------------------------------------------
Base class for all multi-agent verification and synthesis agents.

Local, offline, zero-API dependency:
- Structured reasoning and verdicts are ALWAYS programmatically computed.
- If a local Ollama server is running (e.g. llama3.2), agents generate
  an optional clinician-facing natural-language explanation.
- If Ollama is unavailable, agents fall back cleanly to deterministic templates.
"""
import requests
from typing import Dict, Any, Optional

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:1b"
TIMEOUT_SECONDS = 15


def ollama_available(model: str = DEFAULT_MODEL) -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if r.status_code != 200:
            return False
        tags = [m.get("name", "") for m in r.json().get("models", [])]
        return any(model.split(":")[0] in t for t in tags) or len(tags) > 0
    except Exception:
        return False


def call_local_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 220},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


class BaseAgent:
    name: str = "base_agent"
    use_llm: bool = True

    def __init__(self, model: str = DEFAULT_MODEL, use_llm: bool = True):
        self.model = model
        self.use_llm = use_llm
        self._llm_up: Optional[bool] = None

    def llm_is_up(self) -> bool:
        if self._llm_up is None:
            self._llm_up = self.use_llm and ollama_available(self.model)
        return self._llm_up

    def narrate(self, prompt: str, fallback_text: str) -> str:
        if not self.llm_is_up():
            return fallback_text
        try:
            text = call_local_llm(prompt, self.model)
            return text if text else fallback_text
        except Exception:
            return fallback_text

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
