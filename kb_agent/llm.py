import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-2B")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")


def ask(prompt, max_tokens=32):
    import litellm

    reply = litellm.completion(
        model="openai/" + LLM_MODEL,
        api_base=LLM_API_BASE,
        api_key=LLM_API_KEY,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return (reply.choices[0].message.content or "").strip()
