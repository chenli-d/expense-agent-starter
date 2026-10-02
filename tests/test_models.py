import pytest
from pydantic import ValidationError

from app.models import Claim, Finding, Plan, Report, expected_status, verify_report

REQUIRED = ["check_documents", "estimate_processing"]


def make_finding(rule_id: str, severity: str) -> Finding:
    return Finding(rule_id=rule_id, severity=severity, message="m", source="s")


BLOCK = make_finding("R1", "block")
REMIND = make_finding("R7", "remind")
FLAG = make_finding("R8", "flag")


# --- Plan validators (SPEC section 4) ---

def test_valid_in_scope_plan():
    plan = Plan(in_scope=True, categories=["lodging"], checks=REQUIRED, reason="hotel")
    assert plan.checks == REQUIRED


def test_valid_in_scope_plan_with_policy_check():
    checks = ["check_documents", "check_policy_flags", "estimate_processing"]
    plan = Plan(in_scope=True, categories=["meals"], mentions_alcohol=True, checks=checks, reason="wine")
    assert "check_policy_flags" in plan.checks


def test_valid_out_of_scope_plan():
    plan = Plan(in_scope=False, reason="not travel")
    assert plan.categories == [] and plan.checks == []


@pytest.mark.parametrize("fields", [
    pytest.param(dict(in_scope=False, categories=["meals"]), id="oos-with-categories"),
    pytest.param(dict(in_scope=False, checks=["check_documents"]), id="oos-with-checks"),
    pytest.param(dict(in_scope=True, categories=[], checks=REQUIRED), id="in-scope-no-category"),
    pytest.param(dict(in_scope=True, categories=["lodging"], checks=["check_documents"]),
                 id="missing-estimate_processing"),
    pytest.param(dict(in_scope=True, categories=["lodging"], checks=["estimate_processing"]),
                 id="missing-check_documents"),
    pytest.param(dict(in_scope=True, categories=["lodging"], checks=REQUIRED + ["check_documents"]),
                 id="duplicate-checks"),
    pytest.param(dict(in_scope=True, categories=["hotel"], checks=REQUIRED), id="unknown-category"),
    pytest.param(dict(in_scope=True, categories=["lodging"], checks=REQUIRED + ["approve"]),
                 id="unknown-check"),
])
def test_invalid_plans_are_rejected(fields):
    with pytest.raises(ValidationError):
        Plan(reason="r", **fields)


@pytest.mark.parametrize("reason", ["", "   "])
def test_plan_reason_must_not_be_blank(reason):
    with pytest.raises(ValidationError):
        Plan(in_scope=False, reason=reason)


def test_report_message_must_not_be_blank():
    with pytest.raises(ValidationError):
        Report(status="ready", message=" ")


def test_claim_total_must_not_be_negative():
    with pytest.raises(ValidationError):
        Claim(description="Hotel", total=-1)


# --- expected_status (SPEC section 3 status rule) ---

@pytest.mark.parametrize("findings, status", [
    ([], "ready"),
    ([FLAG], "needs_review"),
    ([REMIND], "needs_review"),
    ([REMIND, FLAG], "needs_review"),
    ([BLOCK], "needs_documents"),
    ([FLAG, BLOCK], "needs_documents"),
])
def test_expected_status(findings, status):
    assert expected_status(findings) == status


# --- verify_report ---

def test_verify_ok_when_status_matches_and_blocks_cited():
    report = Report(status="needs_documents", cited_rules=["R1"], message="m")
    assert verify_report(report, [BLOCK, FLAG]) == []


def test_verify_allows_uncited_non_block_findings():
    report = Report(status="needs_review", cited_rules=[], message="m")
    assert verify_report(report, [REMIND, FLAG]) == []


def test_verify_ok_for_ready_with_no_findings():
    assert verify_report(Report(status="ready", message="m"), []) == []


def test_verify_rejects_wrong_status():
    problems = verify_report(Report(status="ready", cited_rules=["R1"], message="m"), [BLOCK])
    assert len(problems) == 1 and "status" in problems[0]


def test_verify_rejects_unknown_citation():
    report = Report(status="needs_documents", cited_rules=["R1", "R9"], message="m")
    problems = verify_report(report, [BLOCK])
    assert len(problems) == 1 and "R9" in problems[0]


def test_verify_rejects_uncited_block_finding():
    report = Report(status="needs_documents", cited_rules=["R8"], message="m")
    problems = verify_report(report, [BLOCK, FLAG])
    assert len(problems) == 1 and "R1" in problems[0]


def test_verify_reports_every_problem():
    report = Report(status="ready", cited_rules=["R9"], message="m")
    assert len(verify_report(report, [BLOCK])) == 3
