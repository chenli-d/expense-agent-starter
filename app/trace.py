"""Run tracing (SPEC section 6). Shared file: change only via the contract PR.

Every LLM call and tool call becomes one step in Run.steps:
run_id, step, kind (llm|tool), name, args, result (short), ok, ms, usage.
Steps are also appended as JSON lines to a file, but only when run_trace() gets a path.
"""
import functools
import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel

RESULT_MAX_CHARS = 300

_current_run: ContextVar["Run | None"] = ContextVar("current_run", default=None)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def short(value: Any, limit: int = RESULT_MAX_CHARS) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(_jsonable(value), default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class Run:
    def __init__(self, trace_file: Path | None = None):
        self.run_id = uuid.uuid4().hex[:12]
        self.trace_file = trace_file
        self.steps: list[dict] = []

    def record(self, *, kind: str, name: str, args: Any, result: Any, ok: bool,
               ms: float, usage: dict | None = None) -> dict:
        step = {
            "run_id": self.run_id,
            "step": len(self.steps) + 1,
            "kind": kind,
            "name": name,
            "args": _jsonable(args),
            "result": short(result),
            "ok": ok,
            "ms": round(ms, 1),
            "usage": usage or {},
        }
        self.steps.append(step)
        if self.trace_file is not None:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(step, default=str, ensure_ascii=False) + "\n")
        return step

    def tools_called(self) -> list[str]:
        return [s["name"] for s in self.steps if s["kind"] == "tool"]


@contextmanager
def run_trace(trace_file: str | Path | None = None) -> Iterator[Run]:
    """Open a run; steps recorded inside the block go to it.

    trace_file=None keeps steps in memory only. Pass a path (e.g. config.TRACE_FILE) to
    also append them to that file.
    """
    run = Run(Path(trace_file) if trace_file is not None else None)
    token = _current_run.set(run)
    try:
        yield run
    finally:
        _current_run.reset(token)


def current_run() -> Run | None:
    return _current_run.get()


def record(**fields: Any) -> dict | None:
    """Record a step on the active run. No active run -> no-op."""
    run = _current_run.get()
    return run.record(**fields) if run else None


def traced(fn):
    """Decorator for tools: records args, short result, ok and latency as a tool step."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        call_args = {**({"args": list(args)} if args else {}), **kwargs}
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            record(kind="tool", name=fn.__name__, args=call_args,
                   result=f"{type(exc).__name__}: {exc}", ok=False,
                   ms=(time.perf_counter() - start) * 1000)
            raise
        record(kind="tool", name=fn.__name__, args=call_args, result=result, ok=True,
               ms=(time.perf_counter() - start) * 1000)
        return result

    return wrapper
