"""Model access (owner: A). All model calls go through this module.

- LLM:              base class; generate() adds timing and the trace step.
- FakeLLM:          scripted replies (tests).
- RuleBasedFakeLLM: keyword rules that follow SPEC; used by `--fake`, CI and USE_FAKE=1.
- GeminiLLM:        google-genai, JSON output, thinking budget 0, backoff on 429.

Prompt protocol (SPEC section 5): prompts start with "TASK: PLAN" or "TASK: REPORT" and
wrap data as JSON in <claim>...</claim> (a Claim) or <findings>...</findings> (a list of
Finding dicts).
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from app import config
from app.models import Finding, expected_status
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


def extract_tag(prompt: str, tag: str) -> str:
    """Return the text between <tag> and </tag>. Raises ValueError if missing."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", prompt, re.DOTALL)
    if not match:
        raise ValueError(f"prompt has no <{tag}> block")
    return match.group(1)


# Keyword rules for RuleBasedFakeLLM. Matched as whole words, case-insensitive.
_CATEGORY_KEYWORDS = {
    "lodging": r"hotel|motel|lodging|airbnb|nights?",
    "meals": r"meals?|lunch|dinner|breakfast",
    "airfare": r"flights?|airfare|fly|flew|plane",
    "mileage": r"drove|drive|driving|mileage|km|miles",
}
_GUEST_KEYWORDS = r"clients?|guests?|customers?|attendees?"
_ALCOHOL_KEYWORDS = r"alcohol|wine|beer|liquor|cocktails?|whisk(?:e)?y|bar"
_INJECTION_MARKER = re.compile(r"SYSTEM:", re.IGNORECASE)


def _has(pattern: str, text: str) -> bool:
    return re.search(rf"\b(?:{pattern})\b", text, re.IGNORECASE) is not None


class RuleBasedFakeLLM(FakeLLM):
    """Deterministic stand-in that follows SPEC. Passes all gates on evals/cases.jsonl.

    Reads only the data block of the prompt, and ignores claim text after "SYSTEM:".
    """

    name = "rule-based-fake"

    def __init__(self):
        super().__init__(fn=self._reply)

    def _reply(self, prompt: str, step: str) -> dict:
        if prompt.startswith("TASK: PLAN"):
            return self._plan(json.loads(extract_tag(prompt, "claim")))
        if prompt.startswith("TASK: REPORT"):
            return self._report(json.loads(extract_tag(prompt, "findings")))
        raise ValueError("prompt must start with 'TASK: PLAN' or 'TASK: REPORT'")

    @staticmethod
    def _plan(claim: dict) -> dict:
        text = _INJECTION_MARKER.split(claim.get("description", ""), maxsplit=1)[0]
        categories = [c for c, kw in _CATEGORY_KEYWORDS.items() if _has(kw, text)]
        if not categories:
            return {"in_scope": False, "reason": "No travel expense category found."}
        mentions_alcohol = _has(_ALCOHOL_KEYWORDS, text)
        checks = ["check_documents"]
        if mentions_alcohol or float(claim.get("total", 0)) > 2000:
            checks.append("check_policy_flags")
        checks.append("estimate_processing")
        return {
            "in_scope": True,
            "categories": categories,
            "has_guests": _has(_GUEST_KEYWORDS, text),
            "mentions_alcohol": mentions_alcohol,
            "checks": checks,
            "reason": f"Travel expense: {', '.join(categories)}.",
        }

    @staticmethod
    def _report(raw_findings: list[dict]) -> dict:
        findings = [Finding.model_validate(f) for f in raw_findings]
        status = expected_status(findings)
        message = (" ".join(f.message for f in findings) if findings
                   else "No issues found. The claim is ready to submit.")
        return {"status": status, "cited_rules": [f.rule_id for f in findings], "message": message}


class GeminiLLM(LLM):
    """google-genai client. Key from env GEMINI_API_KEY, model id from env MODEL."""

    MAX_ATTEMPTS = 4          # 1 call + 3 retries on 429
    BACKOFF_SECONDS = 2.0     # 2s, 4s, 8s

    def __init__(self, api_key: str | None = None, model: str | None = None):
        model = (model if model is not None else config.MODEL).strip()
        if not model:
            raise ValueError("MODEL is not set. Set MODEL in .env or the environment "
                             "(e.g. MODEL=gemini-2.5-flash-lite).")
        api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        if not api_key.strip():
            raise ValueError("GEMINI_API_KEY is not set. Set it in .env or the environment.")

        from google import genai
        from google.genai import types

        self.name = model
        self._client = genai.Client(api_key=api_key)
        self._config = types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    def _generate(self, prompt: str, step: str) -> LLMResponse:
        from google.genai import errors

        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = self._client.models.generate_content(
                    model=self.name, contents=prompt, config=self._config)
                break
            except errors.APIError as exc:
                if exc.code != 429 or attempt == self.MAX_ATTEMPTS - 1:
                    raise
                time.sleep(self.BACKOFF_SECONDS * 2 ** attempt)

        meta = response.usage_metadata
        usage = {
            "input_tokens": (meta.prompt_token_count or 0) if meta else 0,
            "output_tokens": (meta.candidates_token_count or 0) if meta else 0,
        }
        return LLMResponse(text=response.text or "", usage=usage)
