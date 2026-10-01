# CONTRACT

The agreement between A and B. Build against this, not against each other's code.
Changing anything here needs BOTH people to agree in the PR.

## Shared files (contract PR, both approve changes)
app/models.py, app/config.py, app/trace.py, requirements.txt, SPEC.md, CONTRACT.md, AGENTS.md

## Ownership
| Owner | Files |
|---|---|
| A | app/llm.py, app/planner.py, app/reporter.py, app/pipeline.py, evals/run_eval.py, tests/test_models.py, tests/test_pipeline.py |
| B | app/tools.py, app/ui.py, tests/test_tools.py, .github/workflows/ci.yml |

## app/models.py (pydantic v2)
```python
Category = Literal["lodging", "meals", "airfare", "mileage"]
Check    = Literal["check_documents", "check_policy_flags", "estimate_processing"]
Severity = Literal["block", "remind", "flag"]
Status   = Literal["ready", "needs_documents", "needs_review", "out_of_scope"]

class Claim:    description: str; trip_type: "domestic"|"international" = "domestic";
                total: float (>= 0); attachments: list[str] = []
class Plan:     in_scope: bool; categories: list[Category] = []; has_guests: bool = False;
                mentions_alcohol: bool = False; checks: list[Check] = []; reason: str (non-empty)
                + validators from SPEC section 4
class Finding:  rule_id: str; severity: Severity; message: str; source: str
class Report:   status: Status; cited_rules: list[str] = []; message: str (non-empty); fallback: bool = False
class Outcome:  plan: Plan|None; findings: list[Finding] = []; estimate: dict|None;
                report: Report|None; error: str|None

def expected_status(findings: list[Finding]) -> Status
def verify_report(report: Report, findings: list[Finding]) -> list[str]   # [] means OK
```

## app/tools.py (B), all decorated with @traced, called with keyword arguments
```python
def check_documents(categories: list[str], international: bool, has_guests: bool,
                    attachments: list[str]) -> list[Finding]
def check_policy_flags(mentions_alcohol: bool, total: float, attachments: list[str]) -> list[Finding]
def estimate_processing(categories: list[str]) -> dict   # {"similar_claims": int, "median_days": float|None}
```

## app/pipeline.py (A)
```python
def run_pipeline(llm: LLM, claim: Claim) -> Outcome
```

## app/trace.py (shared)
```python
run_trace(trace_file=None)  # context manager yielding Run; Run.steps: list[dict]; Run.tools_called()
traced                      # decorator for tools
```

## app/llm.py (A)
```python
class LLM:            generate(prompt: str, step: str) -> LLMResponse
class FakeLLM(LLM):   FakeLLM(replies=[...]) or FakeLLM(fn=callable)
class RuleBasedFakeLLM(FakeLLM)
class GeminiLLM(LLM)
```

## Stubs in the contract PR (so nobody waits)
- tools.py: each function returns an empty/neutral result with the right type.
- pipeline.py: returns a canned Outcome (status needs_documents, one R1 finding).
- llm.py: FakeLLM only.
