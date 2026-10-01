"""Plan -> execute -> report -> verify (owner: A).

Contract-PR stub: returns a canned Outcome (needs_documents, one R1 finding).
"""
from app.llm import LLM
from app.models import Claim, Finding, Outcome, Plan, Report


def run_pipeline(llm: LLM, claim: Claim) -> Outcome:
    finding = Finding(
        rule_id="R1",
        severity="block",
        message="Lodging needs an itemized hotel receipt. A card slip alone is not accepted.",
        source="Travel Policy 3.1",
    )
    return Outcome(
        plan=Plan(
            in_scope=True,
            categories=["lodging"],
            checks=["check_documents", "estimate_processing"],
            reason="Stub plan.",
        ),
        findings=[finding],
        estimate=None,
        report=Report(
            status="needs_documents",
            cited_rules=["R1"],
            message="Missing an itemized hotel receipt.",
        ),
        error=None,
    )
