# Reviewing your teammate's PR (10 minutes)

1. Read the PR description. What should change?
2. Files changed: anything outside their ownership? Ask why.
3. Contract: names, fields, signatures match CONTRACT.md?
4. Tests: is there a test for the new behaviour? Does CI pass?
5. Secrets: no keys, no .env.
6. Ask your own agent:
   "Review this diff against CONTRACT.md and SPEC.md. List contract violations,
   missing tests, and risky changes. Do not suggest style nits." (paste `git diff main...<branch>`)
7. Approve, or comment with exactly what to fix.
