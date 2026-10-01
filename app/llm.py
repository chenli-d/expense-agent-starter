"""Model access (owner: A). All model calls go through this module.

Contract-PR stub: LLM base class and FakeLLM only. RuleBasedFakeLLM and GeminiLLM come later.
"""
import json
import time
from dataclasses import dataclass, field
from typing import Callable

from app.trace import record, short


@dataclass
class LLMResponse:
    text: str
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


class LLM:
    """Subclasses implement _generate(). generate() adds timing and the trace step."""

    name = "llm"

    def generate(self, prompt: str, step: str) -> LLMResponse:
        start = time.perf_counter()
        try:
            response = self._generate(prompt, step)
        except Exception as exc:
            record(kind="llm", name=step, args={"model": self.name, "prompt": short(prompt)},
                   result=f"{type(exc).__name__}: {exc}", ok=False,
                   ms=(time.perf_counter() - start) * 1000)
            raise
        record(kind="llm", name=step, args={"model": self.name, "prompt": short(prompt)},
               result=response.text, ok=True, ms=(time.perf_counter() - start) * 1000,
               usage=response.usage)
        return response

    def _generate(self, prompt: str, step: str) -> LLMResponse:
        raise NotImplementedError


class FakeLLM(LLM):
    """Scripted replies for tests: FakeLLM(replies=[...]) or FakeLLM(fn=lambda prompt, step: ...).

    A reply may be a str or a dict (serialized to JSON). Usage is reported as 0 tokens.
    """

    name = "fake"

    def __init__(self, replies: list[str | dict] | None = None,
                 fn: Callable[[str, str], str | dict] | None = None):
        if (replies is None) == (fn is None):
            raise ValueError("pass exactly one of replies or fn")
        self._replies = list(replies) if replies is not None else None
        self._fn = fn
        self.prompts: list[str] = []

    def _generate(self, prompt: str, step: str) -> LLMResponse:
        self.prompts.append(prompt)
        if self._fn is not None:
            reply = self._fn(prompt, step)
        elif self._replies:
            reply = self._replies.pop(0)
        else:
            raise RuntimeError("FakeLLM ran out of scripted replies")
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(text=text)
