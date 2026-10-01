<!-- Claude Code reads this file. Same content as AGENTS.md. -->
# Expense Claim Pre-screen Agent

Read SPEC.md and CONTRACT.md before any task. They are the source of truth.

## Commands
- Test: pytest -q
- Eval offline: python -m evals.run_eval --fake --repeat 3
- Eval real model: python -m evals.run_eval --repeat 3
- App: streamlit run app/ui.py

## Rules
- Only edit files your owner owns (CONTRACT.md "Ownership"), unless told otherwise.
- Never change shared files (models.py, config.py, trace.py, requirements.txt) without saying so first.
- All model calls go through app/llm.py.
- Tests never call a real API.
- Never weaken, skip or delete a test or a gate. Never edit evals/cases.jsonl unless asked.
- Never commit secrets. Keys live in .env or Streamlit secrets.
- Do only the task asked. List every file you changed at the end.
