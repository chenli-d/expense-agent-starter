"""Data contract (CONTRACT.md). Shared file: change only via the contract PR."""
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, model_validator

Category = Literal["lodging", "meals", "airfare", "mileage"]
Check = Literal["check_documents", "check_policy_flags", "estimate_processing"]
Severity = Literal["block", "remind", "flag"]
Status = Literal["ready", "needs_documents", "needs_review", "out_of_scope"]

REQUIRED_CHECKS: tuple[Check, ...] = ("check_documents", "estimate_processing")


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be empty")
    return value


NonEmptyStr = Annotated[str, AfterValidator(_not_blank)]


class Claim(BaseModel):
    # Empty description is allowed here; the pipeline maps it to out_of_scope (SPEC 3.1).
    description: str
    trip_type: Literal["domestic", "international"] = "domestic"
    total: float = Field(ge=0)
    attachments: list[str] = []


class Plan(BaseModel):
    in_scope: bool
    categories: list[Category] = []
    has_guests: bool = False
    mentions_alcohol: bool = False
    checks: list[Check] = []
    reason: NonEmptyStr

    @model_validator(mode="after")
    def _check_plan_rules(self) -> "Plan":
        # SPEC section 4.
        if not self.in_scope:
            if self.categories or self.checks:
                raise ValueError("out-of-scope plan must have empty categories and checks")
            return self
        if not self.categories:
            raise ValueError("in-scope plan needs at least one category")
        missing = [c for c in REQUIRED_CHECKS if c not in self.checks]
        if missing:
            raise ValueError(f"in-scope plan is missing checks: {missing}")
        if len(set(self.checks)) != len(self.checks):
            raise ValueError("checks must not contain duplicates")
        return self


class Finding(BaseModel):
    rule_id: str
    severity: Severity
    message: str
    source: str


class Report(BaseModel):
    status: Status
    cited_rules: list[str] = []
    message: NonEmptyStr
    fallback: bool = False


class Outcome(BaseModel):
    plan: Plan | None = None
    findings: list[Finding] = []
    estimate: dict | None = None
    report: Report | None = None
    error: str | None = None


def expected_status(findings: list[Finding]) -> Status:
    """Status rule from SPEC section 3. Out-of-scope is decided by the pipeline, not here."""
    if any(f.severity == "block" for f in findings):
        return "needs_documents"
    if findings:
        return "needs_review"
    return "ready"


def verify_report(report: Report, findings: list[Finding]) -> list[str]:
    """Check a report against the evidence. Returns problems; [] means OK."""
    problems: list[str] = []
    want = expected_status(findings)
    if report.status != want:
        problems.append(f"status is {report.status!r}, evidence says {want!r}")

    finding_ids = {f.rule_id for f in findings}
    cited = set(report.cited_rules)
    unknown = sorted(cited - finding_ids)
    if unknown:
        problems.append(f"cites rules not in findings: {unknown}")
    # Only blocking findings must be cited; remind/flag findings may be left out.
    block_ids = {f.rule_id for f in findings if f.severity == "block"}
    uncited = sorted(block_ids - cited)
    if uncited:
        problems.append(f"does not cite blocking findings: {uncited}")
    return problems
