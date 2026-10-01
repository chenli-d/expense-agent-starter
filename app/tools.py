"""Deterministic tools (owner: B). Contract-PR stubs: neutral results with the right types."""
from app.models import Finding
from app.trace import traced


@traced
def check_documents(categories: list[str], international: bool, has_guests: bool,
                    attachments: list[str]) -> list[Finding]:
    return []


@traced
def check_policy_flags(mentions_alcohol: bool, total: float, attachments: list[str]) -> list[Finding]:
    return []


@traced
def estimate_processing(categories: list[str]) -> dict:
    return {"similar_claims": 0, "median_days": None}
