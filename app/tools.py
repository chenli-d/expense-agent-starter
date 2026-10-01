"""Deterministic expense checks backed by policy rules and claim history."""
import csv
import json
from pathlib import Path
from statistics import median

from app.models import Finding
from app.trace import traced


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _rules() -> list[dict]:
    with (DATA_DIR / "policy_rules.json").open(encoding="utf-8") as file:
        return json.load(file)


def _finding(rule: dict) -> Finding:
    return Finding(rule_id=rule["id"], severity=rule["severity"],
                   message=rule["message"], source=rule["source"])


@traced
def check_documents(categories: list[str], international: bool, has_guests: bool,
                    attachments: list[str]) -> list[Finding]:
    findings = []
    for rule in _rules():
        if rule["kind"] != "document":
            continue
        conditions = rule["when"]
        if "category" in conditions and conditions["category"] not in categories:
            continue
        if "international" in conditions and international != conditions["international"]:
            continue
        if ("guests_with_meals" in conditions
                and (has_guests and "meals" in categories) != conditions["guests_with_meals"]):
            continue
        if rule["requires"] not in attachments:
            findings.append(_finding(rule))
    return findings


@traced
def check_policy_flags(mentions_alcohol: bool, total: float, attachments: list[str]) -> list[Finding]:
    findings = []
    for rule in _rules():
        if rule["kind"] != "policy":
            continue
        conditions = rule["when"]
        if "total_over" in conditions and total <= conditions["total_over"]:
            continue
        if ("mentions_alcohol" in conditions
                and mentions_alcohol != conditions["mentions_alcohol"]):
            continue
        if rule["requires"] is None or rule["requires"] not in attachments:
            findings.append(_finding(rule))
    return findings


@traced
def estimate_processing(categories: list[str]) -> dict:
    """Use historical claims with the same category set, regardless of order."""
    requested = set(categories)
    with (DATA_DIR / "history.csv").open(encoding="utf-8", newline="") as file:
        days = [float(row["days"]) for row in csv.DictReader(file)
                if set(row["categories"].split(";")) == requested]
    return {"similar_claims": len(days), "median_days": median(days) if days else None}
