"""Minimal client for the local Ollama HTTP API."""
import requests

from . import config


def chat(model: str, messages: list[dict], temperature: float = 0.0, num_predict: int = 400,
         num_ctx: int = 4096, json_format: bool = False, timeout: int = 600) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "seed": config.SEED,
                    "num_predict": num_predict, "num_ctx": num_ctx, "num_gpu": config.OLLAMA_NUM_GPU},
    }
    if json_format:
        payload["format"] = "json"
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def is_available(model: str) -> bool:
    try:
        tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5).json()
    except requests.RequestException:
        return False
    return any(m["name"] == model or m["name"].split(":")[0] == model for m in tags.get("models", []))
