import json
import time
from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncOpenAI, RateLimitError, APIError
from app.config import settings


class FreeLLMClient:
    """
    Unified OpenAI-compatible async client supporting free tiers:
    Groq, Google Gemini, OpenRouter, Mistral, and local Ollama.
    Includes automatic fallback on free-tier 429 rate limits.
    """

    def __init__(self):
        self.primary_client = AsyncOpenAI(
            api_key=settings.llm_api_key or "missing-key",
            base_url=settings.llm_base_url,
        )
        self.primary_model = settings.llm_model

        self.fallback_client: Optional[AsyncOpenAI] = None
        if settings.fallback_api_key:
            self.fallback_client = AsyncOpenAI(
                api_key=settings.fallback_api_key,
                base_url=settings.fallback_base_url,
            )
        self.fallback_model = settings.fallback_model

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        json_mode: bool = False,
        max_tokens: int = 2048,
    ) -> Tuple[str, Dict[str, Any]]:
        start = time.perf_counter()
        kwargs: Dict[str, Any] = {
            "model": self.primary_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = await self.primary_client.chat.completions.create(**kwargs)
            model_used = self.primary_model
        except (RateLimitError, APIError) as e:
            if self.fallback_client:
                kwargs["model"] = self.fallback_model
                resp = await self.fallback_client.chat.completions.create(**kwargs)
                model_used = self.fallback_model
            else:
                raise RuntimeError(f"Primary free LLM failed ({e}) and no fallback configured.")

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        content = resp.choices[0].message.content or ""
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        meta = {
            "model": model_used,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency_ms,
            "cost_usd": 0.0,  # Free tier inference
        }
        return content, meta

    async def generate_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raw, meta = await self.chat_completion(
            messages=messages, temperature=temperature, json_mode=True
        )
        # Clean markdown fences if model emits them despite json_mode
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        return json.loads(cleaned), meta


llm_client = FreeLLMClient()
