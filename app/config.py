"""Paths, model settings and limits. Shared file: change only via the contract PR."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    # Does not override variables already set in the environment.
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

RULES_FILE = ROOT / "data" / "policy_rules.json"
HISTORY_FILE = ROOT / "data" / "history.csv"
TRACE_FILE = ROOT / "traces.jsonl"

# No default model: GeminiLLM must raise a clear error naming MODEL when this is empty.
MODEL = os.getenv("MODEL", "").strip()
PRICE_IN_PER_M = float(os.getenv("PRICE_IN_PER_M") or 0.10)
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUT_PER_M") or 0.40)

MAX_RETRIES = 1
MAX_STEPS = 8

# SPEC section 7: every gate must be 1.0, else run_eval exits 1.
GATES = {
    "schema_valid_rate": 1.0,
    "tool_check_ok_rate": 1.0,
    "adversarial_pass_k": 1.0,
    "out_of_scope_recall": 1.0,
}
