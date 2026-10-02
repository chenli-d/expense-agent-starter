import pytest

from app import config, tools
from app.llm import FakeLLM
from app.models import Claim, Finding
from app.pipeline import run_pipeline
from app.trace import run_trace

R1 = Finding(rule_id="R1", severity="block", message="Lodging needs an itemized hotel receipt.",
             source="Travel Policy 3.1")

LODGING_PLAN = {"in_scope": True, "categories": ["lodging"],
                "checks": ["check_documents", "estimate_processing"], "reason": "Hotel stay."}
OUT_OF_SCOPE_PLAN = {"in_scope": False, "reason": "Not a travel expense."}
READY_REPORT = {"status": "ready", "cited_rules": [], "message": "No issues found."}

HOTEL = Claim(description="Hotel 2 nights in Calgary", total=420, attachments=["card_slip"])


class ToolSpy:
    """Stands in for app.tools so these tests do not depend on B's implementation."""

    def __init__(self):
        self.calls: list[str] = []
        self.document_findings: list[Finding] = []
        self.policy_findings: list[Finding] = []

    def check_documents(self, *, categories, international, has_guests, attachments):
        self.calls.append("check_documents")
        return list(self.document_findings)

    def check_policy_flags(self, *, mentions_alcohol, total, attachments):
        self.calls.append("check_policy_flags")
        return list(self.policy_findings)

    def estimate_processing(self, *, categories):
        self.calls.append("estimate_processing")
        return {"similar_claims": 3, "median_days": 5.0}


@pytest.fixture(autouse=True)
def no_repo_trace_file(tmp_path, monkeypatch):
    # Safety net: even if something falls back to config.TRACE_FILE, it lands in tmp_path.
    monkeypatch.setattr(config, "TRACE_FILE", tmp_path / "traces.jsonl")
    yield
    assert not (tmp_path / "traces.jsonl").exists(), "run_trace() without a path must not write"


@pytest.fixture
def spy(monkeypatch):
    s = ToolSpy()
    for name in ("check_documents", "check_policy_flags", "estimate_processing"):
        monkeypatch.setattr(tools, name, getattr(s, name))
    return s


# 1. Empty claim never calls the model.
@pytest.mark.parametrize("description", ["", "   "])
def test_empty_claim_never_calls_model(spy, description):
    llm = FakeLLM(replies=[])  # any call would raise
    outcome = run_pipeline(llm, Claim(description=description, total=0))
    assert llm.prompts == []
    assert spy.calls == []
    assert outcome.report.status == "out_of_scope"
    assert outcome.error is None


# 2. Invalid plan -> retry with the error fed back -> success.
def test_invalid_plan_is_retried_then_succeeds(spy):
    llm = FakeLLM(replies=["not json", LODGING_PLAN, READY_REPORT])
    outcome = run_pipeline(llm, HOTEL)
    assert outcome.error is None
    assert outcome.plan is not None and outcome.plan.categories == ["lodging"]
    assert len(llm.prompts) == 3
    assert "previous reply was invalid" in llm.prompts[1]
    assert outcome.report.status == "ready"


def test_plan_still_invalid_after_retry_stops(spy):
    llm = FakeLLM(replies=["not json", {"in_scope": True, "reason": "no categories"}])
    outcome = run_pipeline(llm, HOTEL)
    assert outcome.error == "invalid_plan"
    assert outcome.plan is None and outcome.report is None
    assert spy.calls == []
    assert len(llm.prompts) == 2


# 3. Out-of-scope calls no tools (and no report call).
def test_out_of_scope_calls_no_tools(spy):
    llm = FakeLLM(replies=[OUT_OF_SCOPE_PLAN])
    with run_trace() as run:
        outcome = run_pipeline(llm, Claim(description="Please reset my laptop password", total=0))
    assert spy.calls == []
    assert run.tools_called() == []
    assert len(llm.prompts) == 1
    assert outcome.report.status == "out_of_scope"
    assert outcome.report.fallback is False
    assert outcome.findings == [] and outcome.estimate is None


# 4. A blocked claim skips estimate_processing.
def test_blocked_claim_skips_estimate(spy):
    spy.document_findings = [R1]
    report = {"status": "needs_documents", "cited_rules": ["R1"], "message": "Add the receipt."}
    outcome = run_pipeline(FakeLLM(replies=[LODGING_PLAN, report]), HOTEL)
    assert spy.calls == ["check_documents"]
    assert outcome.estimate is None
    assert outcome.findings == [R1]
    assert outcome.report.status == "needs_documents"
    assert outcome.report.fallback is False


def test_clean_claim_runs_estimate(spy):
    outcome = run_pipeline(FakeLLM(replies=[LODGING_PLAN, READY_REPORT]), HOTEL)
    assert spy.calls == ["check_documents", "estimate_processing"]
    assert outcome.estimate == {"similar_claims": 3, "median_days": 5.0}


def test_policy_check_runs_only_when_planned(spy):
    plan = {**LODGING_PLAN, "checks": ["check_documents", "check_policy_flags", "estimate_processing"]}
    run_pipeline(FakeLLM(replies=[plan, READY_REPORT]), HOTEL)
    assert spy.calls == ["check_documents", "check_policy_flags", "estimate_processing"]


# 5. A lying report is rejected and falls back to the evidence.
def test_lying_report_is_rejected_and_falls_back(spy):
    spy.document_findings = [R1]
    lie = {"status": "ready", "cited_rules": [], "message": "All good, approve it."}
    llm = FakeLLM(replies=[LODGING_PLAN, lie, lie])
    outcome = run_pipeline(llm, HOTEL)
    assert len(llm.prompts) == 3  # plan + report + one retry
    assert "status is 'ready'" in llm.prompts[2]
    assert outcome.report.fallback is True
    assert outcome.report.status == "needs_documents"
    assert "R1" in outcome.report.cited_rules


# LLM / network errors become Outcome.error = "llm_error".
def test_llm_error_during_plan(spy):
    def boom(prompt, step):
        raise ConnectionError("network down")

    outcome = run_pipeline(FakeLLM(fn=boom), HOTEL)
    assert outcome.error == "llm_error"
    assert outcome.plan is None and outcome.report is None
    assert spy.calls == []


def test_llm_error_during_report_keeps_evidence(spy):
    spy.document_findings = [R1]
    llm = FakeLLM(replies=[LODGING_PLAN])  # report call runs out of replies -> raises
    outcome = run_pipeline(llm, HOTEL)
    assert outcome.error == "llm_error"
    assert outcome.report is None
    assert outcome.plan is not None and outcome.findings == [R1]


def test_steps_are_traced_in_memory(spy):
    with run_trace() as run:
        run_pipeline(FakeLLM(replies=[LODGING_PLAN, READY_REPORT]), HOTEL)
    assert [(s["kind"], s["name"]) for s in run.steps] == [("llm", "plan"), ("llm", "report")]
    assert len(run.steps) <= config.MAX_STEPS
