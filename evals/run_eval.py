"""Offline / real-model evaluation (owner: A). SPEC section 7.

    python -m evals.run_eval --fake --repeat 3   # RuleBasedFakeLLM, no API key
    python -m evals.run_eval --repeat 3          # GeminiLLM (GEMINI_API_KEY, MODEL)

A run is correct when status AND the set of blocking rule ids both match the case.
Exits 1 if any gate in config.GATES is not met.
"""
import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from app import config
from app.llm import LLM, GeminiLLM, RuleBasedFakeLLM
from app.models import Claim
from app.pipeline import run_pipeline
from app.trace import run_trace

CASES_FILE = Path(__file__).resolve().parent / "cases.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_cases(path: Path = CASES_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_tools(case: dict, tools_called: list[str], steps: int) -> list[str]:
    """Loose tool checks from SPEC section 7 (order is not checked)."""
    problems = []
    expected = case["expected_status"]
    if expected == "out_of_scope":
        if tools_called:
            problems.append(f"out_of_scope called tools {tools_called}")
    else:
        if "check_documents" not in tools_called:
            problems.append("check_documents not called")
        if expected != "needs_documents" and "estimate_processing" not in tools_called:
            problems.append("estimate_processing not called")
    if case["needs_policy_check"] and "check_policy_flags" not in tools_called:
        problems.append("check_policy_flags not called")
    if steps > config.MAX_STEPS:
        problems.append(f"{steps} steps > MAX_STEPS={config.MAX_STEPS}")
    return problems


def run_case(llm: LLM, case: dict) -> dict:
    claim = Claim.model_validate(case["claim"])
    with run_trace(config.TRACE_FILE) as run:
        start = time.perf_counter()
        outcome = run_pipeline(llm, claim)
        ms = (time.perf_counter() - start) * 1000

    status = outcome.report.status if outcome.report else "error"
    blocks = sorted({f.rule_id for f in outcome.findings if f.severity == "block"})
    tool_problems = check_tools(case, run.tools_called(), len(run.steps))
    usage = Counter()
    for step in run.steps:
        if step["kind"] == "llm":
            usage.update({k: v for k, v in step["usage"].items() if isinstance(v, int)})
    return {
        "case_id": case["id"],
        "run_id": run.run_id,
        "status": status,
        "blocks": blocks,
        "correct": status == case["expected_status"] and blocks == sorted(case["expected_blocks"]),
        "schema_valid": outcome.error is None and outcome.report is not None,
        "error": outcome.error,
        "fallback": bool(outcome.report and outcome.report.fallback),
        "tools_called": run.tools_called(),
        "steps": len(run.steps),
        "tool_problems": tool_problems,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "ms": round(ms, 1),
    }


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def summarize(cases: list[dict], runs: list[dict], k: int) -> dict:
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_case[r["case_id"]].append(r)
    case_by_id = {c["id"]: c for c in cases}

    def rate(items: list[bool]) -> float:
        return round(sum(items) / len(items), 4) if items else 1.0

    # pass@1: expected single-run accuracy. pass^k: case correct in all k runs.
    pass_1 = rate([r["correct"] for r in runs])
    pass_k = rate([all(r["correct"] for r in rs) for rs in by_case.values()])

    # Majority baseline: always predict the most common expected status, no blocks.
    majority_status, _ = Counter(c["expected_status"] for c in cases).most_common(1)[0]
    majority_baseline = rate([c["expected_status"] == majority_status and not c["expected_blocks"]
                              for c in cases])

    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in runs:
        confusion[case_by_id[r["case_id"]]["expected_status"]][r["status"]] += 1

    flaky = sorted(cid for cid, rs in by_case.items()
                   if len({(r["status"], tuple(r["blocks"])) for r in rs}) > 1)

    adversarial = [all(r["correct"] for r in rs) for cid, rs in by_case.items()
                   if case_by_id[cid]["type"] == "adversarial"]
    oos_runs = [r["status"] == "out_of_scope" for r in runs
                if case_by_id[r["case_id"]]["expected_status"] == "out_of_scope"]

    metrics = {
        "schema_valid_rate": rate([r["schema_valid"] for r in runs]),
        "tool_check_ok_rate": rate([not r["tool_problems"] for r in runs]),
        "adversarial_pass_k": rate(adversarial),
        "out_of_scope_recall": rate(oos_runs),
    }
    gates = {name: {"value": metrics[name], "threshold": threshold,
                    "passed": metrics[name] >= threshold}
             for name, threshold in config.GATES.items()}

    input_tokens = sum(r["input_tokens"] for r in runs)
    output_tokens = sum(r["output_tokens"] for r in runs)
    cost = (input_tokens * config.PRICE_IN_PER_M + output_tokens * config.PRICE_OUT_PER_M) / 1e6
    latencies = [r["ms"] for r in runs]

    return {
        "cases": len(cases),
        "repeat": k,
        "runs": len(runs),
        "accuracy": {"pass@1": pass_1, f"pass^{k}": pass_k, "majority_baseline": majority_baseline,
                     "majority_status": majority_status},
        "metrics": metrics,
        "gates": gates,
        "status_confusion": {exp: dict(pred) for exp, pred in sorted(confusion.items())},
        "flaky_cases": flaky,
        "incorrect_cases": sorted({r["case_id"] for r in runs if not r["correct"]}),
        "tool_check_failures": {r["case_id"]: r["tool_problems"] for r in runs if r["tool_problems"]},
        "errors": dict(Counter(r["error"] for r in runs if r["error"])),
        "report_fallbacks": sum(r["fallback"] for r in runs),
        "tokens": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
        "cost_usd": round(cost, 6),
        "latency_ms": {"avg": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
                       "p95": percentile(latencies, 95),
                       "max": max(latencies, default=0.0)},
    }


def print_report(summary: dict, model: str) -> None:
    k = summary["repeat"]
    acc = summary["accuracy"]
    lat = summary["latency_ms"]
    tok = summary["tokens"]
    lines = [
        f"=== Eval: model={model}  cases={summary['cases']}  repeat={k}  runs={summary['runs']} ===",
        f"Accuracy   pass@1={acc['pass@1']:.3f}  pass^{k}={acc[f'pass^{k}']:.3f}  "
        f"majority baseline={acc['majority_baseline']:.3f} (always '{acc['majority_status']}')",
        "Status confusion (expected -> predicted: count):",
    ]
    for expected, predicted in summary["status_confusion"].items():
        cells = ", ".join(f"{p}: {n}" for p, n in sorted(predicted.items()))
        lines.append(f"  {expected:16} -> {cells}")
    lines += [
        f"Incorrect cases: {summary['incorrect_cases'] or 'none'}",
        f"Flaky cases:     {summary['flaky_cases'] or 'none'}",
        f"Tool check failures: {summary['tool_check_failures'] or 'none'}",
        f"Errors:          {summary['errors'] or 'none'}",
        f"Report fallbacks: {summary['report_fallbacks']}",
        f"Tokens   in={tok['input']}  out={tok['output']}  total={tok['total']}  "
        f"cost=${summary['cost_usd']:.6f}",
        f"Latency  avg={lat['avg']:.1f}ms  p95={lat['p95']:.1f}ms  max={lat['max']:.1f}ms",
        "Gates:",
    ]
    for name, g in summary["gates"].items():
        mark = "PASS" if g["passed"] else "FAIL"
        lines.append(f"  [{mark}] {name} = {g['value']:.3f} (needs >= {g['threshold']})")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the pre-screen agent on evals/cases.jsonl")
    parser.add_argument("--fake", action="store_true", help="use RuleBasedFakeLLM (no API calls)")
    parser.add_argument("--repeat", type=int, default=1, metavar="k", help="runs per case (default 1)")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    llm: LLM = RuleBasedFakeLLM() if args.fake else GeminiLLM()
    cases = load_cases()
    runs = [run_case(llm, case) for case in cases for _ in range(args.repeat)]
    summary = summarize(cases, runs, args.repeat)
    print_report(summary, llm.name)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": llm.name, "fake": args.fake, "summary": summary, "runs": runs}, f, indent=2)
    print(f"Saved {out}")

    return 0 if all(g["passed"] for g in summary["gates"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
