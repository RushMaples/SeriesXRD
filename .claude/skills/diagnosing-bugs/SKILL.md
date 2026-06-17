---
name: diagnosing-bugs
description: Systematically diagnose hard-to-find bugs and performance regressions using a six-phase feedback-loop methodology.
disable-model-invocation: true
---

# Diagnosing Bugs

A disciplined six-phase approach for debugging difficult issues and performance problems.

## Core Philosophy

**Building a tight feedback loop is the critical skill.** "If you have a tight pass/fail signal for the bug — one that goes red on _this_ bug — you will find the cause." Jumping straight to a hypothesis without a working feedback loop is the exact failure this skill prevents.

Read `CONTEXT.md` (if it exists) before diving in — the domain vocabulary will help you name hypotheses and findings clearly.

## Phase 1: Build a Feedback Loop

Create a reliable way to reproduce the bug. Escalating strategies:

1. Write a failing test that isolates the symptom
2. Write a script that reproduces it end-to-end
3. Use the REPL / interactive session to probe the state
4. Add targeted logging to a running process
5. Use a debugger with breakpoints at the suspected entry
6. Capture the failing network/IO traffic
7. Add property-based fuzzing to surface the edge case
8. Differential comparison (before vs. after a commit)
9. Bisect commits to find the introduction point
10. Minimal reproduction case (strip to the smallest failing input)

Treat the loop itself as a product: optimize for speed, signal clarity, and determinism. A slow loop that takes 30 seconds per iteration costs you an hour for 120 cycles.

## Phase 2: Reproduce and Minimise

Run the loop to confirm the actual failure matches the user's report. Then systematically shrink the scenario — remove non-essential elements one at a time until you have the smallest case that still goes red.

A minimized reproduction is dramatically easier to reason about.

## Phase 3: Hypothesize

Generate **3–5 ranked, falsifiable hypotheses** before testing any of them. Each hypothesis should predict a specific observable change (not just "it might be X").

Review with a domain expert if the area is unfamiliar. Bad hypothesis: "the database is slow." Good hypothesis: "query plan regression on the `orders` table after the index drop — should show up in EXPLAIN ANALYZE."

## Phase 4: Instrument

Test predictions with targeted debugging:

- **Prefer debuggers over log dumps** — breakpoints give you the full state at a point in time; logs give you what you thought to ask for
- Tag all temporary debug output with a unique prefix for easy cleanup
- For performance issues: measure, don't log — use profilers and timers, not console output
- Test the highest-ranked hypothesis first; eliminate before moving down the list

## Phase 5: Fix and Write Regression Tests

Before applying the fix:

1. Write a test at the correct architectural seam that exercises the real bug pattern
2. Confirm the test goes RED against the unfixed code
3. Apply the fix
4. Confirm the test goes GREEN
5. Validate against both the minimized case and the original scenario

The regression test is the permanent artifact. The fix is only useful if the test keeps it from regressing.

## Phase 6: Cleanup and Post-Mortem

- Remove all temporary instrumentation (tagged debug output, temporary scripts)
- Verify the original reported scenario no longer reproduces
- Document findings in the commit message: what the root cause was, how it was found, what the fix does
- Flag any architectural improvement implied by the bug for follow-up (don't do it now — it's out of scope)
