"""Plan -> execute -> report -> verify (owner: A). SPEC section 3.

The LLM plans and writes the report; code runs the tools, decides the status and verifies.
Bounded by construction: at most 2 plan calls + 3 tools + 2 report calls = 7 steps,
which is within config.MAX_STEPS.

The caller opens run_trace() if it wants the steps; the pipeline only records into it.
"""
import logging

from app import tools
from app.llm import LLM
from app.models import Claim, Finding, Outcome, Plan, Report
from app.planner import make_plan
from app.reporter import write_report

logger = logging.getLogger(__name__)

LLM_ERROR = "llm_error"

EMPTY_MESSAGE = "The claim has no description, so it cannot be pre-screened."
OUT_OF_SCOPE_MESSAGE = "This does not look like a travel expense claim, so it was not pre-screened."


def _out_of_scope(plan: Plan | None, message: str) -> Outcome:
    # Built by code, not the LLM; out_of_scope reports never go through write_report.
    return Outcome(plan=plan, report=Report(status="out_of_scope", message=message))


def run_pipeline(llm: LLM, claim: Claim) -> Outcome:
    # 1. Empty description: out of scope, no model call.
    if not claim.description.strip():
        return _out_of_scope(None, EMPTY_MESSAGE)

    # 2. PLAN (LLM). make_plan retries once on validation errors.
    # Exceptions here are LLM/transport failures (validation errors are handled inside).
    try:
        plan, error = make_plan(llm, claim)
    except Exception:
        logger.exception("PLAN step failed; returning error=%s", LLM_ERROR)
        return Outcome(error=LLM_ERROR)
    if plan is None:
        return Outcome(error=error)

    # 3. Out of scope: no tools, no report call.
    if not plan.in_scope:
        return _out_of_scope(plan, OUT_OF_SCOPE_MESSAGE)

    # 4. EXECUTE (code), fixed order. Tools are called through the module so tests
    # can monkeypatch them.
    findings: list[Finding] = list(tools.check_documents(
        categories=plan.categories,
        international=claim.trip_type == "international",
        has_guests=plan.has_guests,
        attachments=claim.attachments,
    ))
    if "check_policy_flags" in plan.checks:
        findings += tools.check_policy_flags(
            mentions_alcohol=plan.mentions_alcohol,
            total=claim.total,
            attachments=claim.attachments,
        )
    estimate = None
    blocked = any(f.severity == "block" for f in findings)
    if "estimate_processing" in plan.checks and not blocked:
        estimate = tools.estimate_processing(categories=plan.categories)

    # 5-6. REPORT (LLM) + VERIFY (code). write_report never returns an unverified report.
    try:
        report = write_report(llm, findings)
    except Exception:
        logger.exception("REPORT step failed; returning error=%s", LLM_ERROR)
        return Outcome(plan=plan, findings=findings, estimate=estimate, error=LLM_ERROR)

    return Outcome(plan=plan, findings=findings, estimate=estimate, report=report)
