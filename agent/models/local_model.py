"""Client pour llama-server (Qwen 3.5 9B via llama.cpp).

Deux modes :
- chat() : reponse texte libre (backward compat, subagents)
- chat_with_tools() : tool calling natif OpenAI-compatible (--jinja)
"""

from __future__ import annotations

import json
import httpx
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "model.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class LocalModel:
    def __init__(self, config: dict | None = None):
        cfg = config or _load_config()
        self.base_url = cfg["llama_server"]["base_url"]
        self.gen = cfg["generation"]
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=600)

    async def chat(self, messages: list[dict]) -> str:
        """Appel LLM sans tools. Retourne le contenu texte brut."""
        payload = {
            "model": "qwen3.5-9b",
            "messages": messages,
            "temperature": self.gen["temperature"],
            "top_p": self.gen["top_p"],
            "top_k": self.gen.get("top_k", 20),
            "min_p": self.gen.get("min_p", 0.0),
            "presence_penalty": self.gen.get("presence_penalty", 1.5),
            "max_tokens": self.gen["max_tokens"],
            "stop": self.gen.get("stop", []),
            "stream": False,
        }
        resp = await self.client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content", "").strip()
        self._last_reasoning = msg.get("reasoning_content", "")

        # Si content vide mais reasoning existe, extraire la reponse
        if not content and self._last_reasoning:
            reasoning = self._last_reasoning.strip()
            # Chercher un JSON dans un code block
            if "```" in reasoning:
                for part in reasoning.split("```"):
                    clean = part.strip()
                    if clean.startswith("json"):
                        clean = clean[4:].strip()
                    if clean.startswith("{"):
                        try:
                            json.loads(clean)
                            content = clean
                            break
                        except json.JSONDecodeError:
                            continue
            # Chercher un JSON brut
            if not content:
                start = reasoning.rfind("{")
                if start != -1:
                    for end in range(len(reasoning), start, -1):
                        if reasoning[end - 1] == "}":
                            try:
                                json.loads(reasoning[start:end])
                                content = reasoning[start:end]
                                break
                            except json.JSONDecodeError:
                                continue
            if not content:
                content = reasoning[-500:]

        finish = data["choices"][0].get("finish_reason", "")
        if finish == "length" and not content:
            content = "[Reponse tronquee - contexte insuffisant]"

        return content

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> dict:
        """Appel LLM avec tool calling natif (requiert --jinja).

        Retourne :
        {
            "content": str | None,       # reponse texte (si pas de tool call)
            "tool_calls": list | None,    # liste de tool calls
            "reasoning": str | None,      # raisonnement interne
        }

        Chaque tool_call :
        {
            "id": str,
            "function": {"name": str, "arguments": dict}
        }
        """
        payload = {
            "model": "qwen3.5-9b",
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": self.gen["temperature"],
            "top_p": self.gen["top_p"],
            "top_k": self.gen.get("top_k", 20),
            "min_p": self.gen.get("min_p", 0.0),
            "presence_penalty": self.gen.get("presence_penalty", 1.5),
            "max_tokens": self.gen["max_tokens"],
            "stream": False,
        }
        resp = await self.client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]

        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()

        # Si content vide, pas de tool_calls, et reasoning rempli :
        # le modele a mis sa reponse dans le thinking (comportement Qwen 3.5)
        raw_calls = msg.get("tool_calls")
        if not content and not raw_calls and reasoning:
            # Extraire la derniere partie utile du reasoning comme reponse
            content = reasoning[-500:]

        finish = data["choices"][0].get("finish_reason", "")

        result = {
            "content": content or None,
            "tool_calls": None,
            "reasoning": reasoning or None,
            "truncated": finish == "length",
        }

        # Parser les tool_calls
        if raw_calls:
            parsed_calls = []
            for tc in raw_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                # Parse arguments de JSON string a dict
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {"_raw": args_str}
                parsed_calls.append({
                    "id": tc.get("id", ""),
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": args,
                    }
                })
            result["tool_calls"] = parsed_calls

        return result

    async def chat_stream(self, messages: list[dict]):
        """Streaming (inchange)."""
        payload = {
            "model": "qwen3.5-9b",
            "messages": messages,
            "temperature": self.gen["temperature"],
            "top_p": self.gen["top_p"],
            "max_tokens": self.gen["max_tokens"],
            "stop": self.gen["stop"],
            "stream": True,
        }
        async with self.client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk.strip() == "[DONE]":
                    break
                data = json.loads(chunk)
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get("/models")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    async def close(self):
        await self.client.aclose()
