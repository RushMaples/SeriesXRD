---
name: grill-with-docs
description: Sharpen a plan or idea through intensive questioning while simultaneously generating CONTEXT.md domain glossary and ADRs from the codebase.
disable-model-invocation: true
---

# Grill With Docs

Conduct an intensive grilling session that challenges your plan against the existing domain model, sharpens terminology, and updates `CONTEXT.md` and ADRs inline as decisions land.

Run a `/grilling` session using the `/domain-modeling` skill.

## What this produces

- Clarified, battle-tested plan or design
- Updated `CONTEXT.md` with any new domain terms coined during the session
- ADRs for decisions that are hard to reverse, surprising without context, and represent genuine trade-offs
- A shared vocabulary that the rest of the engineering skills can read

## Process

1. Read `CONTEXT.md` (if it exists) and any ADRs in `docs/adr/` before starting — understand the domain language already established
2. Run the grilling loop: one question at a time, wait for an answer, follow up before moving on. Never dump a list of questions.
3. Inline side effects as decisions crystallize:
   - New term coined? Add to `CONTEXT.md` immediately
   - Fuzzy term sharpened? Update `CONTEXT.md` right there
   - Hard-to-reverse architectural decision made? Offer an ADR
4. At the end, summarize what was decided and what changed in the docs

## Grilling principles

- Ask one question at a time
- Follow up before moving on — surface the assumptions behind the first answer
- Challenge vague language ("what do you mean by 'account' here?")
- Stress-test with edge cases and failure modes
- Stop when the plan is sharp enough to hand to `/to-prd` or `/implement`
