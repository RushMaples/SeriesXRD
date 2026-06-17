---
name: domain-modeling
description: Actively build and maintain the project's domain model — challenge vague language, stress-test with scenarios, update CONTEXT.md and ADRs inline as decisions land.
disable-model-invocation: false
---

# Domain Modeling

Actively build and maintain the project's domain model. Don't passively read existing docs — **challenge, sharpen, and update** as the conversation progresses.

## Core responsibilities

**Challenge and sharpen language.** When the user uses a vague or conflicting term, propose a precise alternative. For example: "You said 'account' — do you mean the Customer or the User entity? They're different things in your model."

**Stress-test with scenarios.** Use concrete edge cases to expose fuzzy thinking and force precision around concept boundaries. "What happens if the user cancels part of an Order — is that a partial cancellation or a new Order?"

**Cross-reference code.** Surface contradictions between stated behavior and actual implementation. "If your code cancels entire Orders, but you've said partial cancellation is possible — which is correct?"

**Update immediately.** Don't batch updates. The moment a term crystallizes, update `CONTEXT.md`. The moment an architectural decision is made, offer an ADR.

## Documentation structure

**Single-context repos**: one `CONTEXT.md` + `docs/adr/` at the repo root.

**Multi-context repos** (monorepos): a `CONTEXT-MAP.md` at the root pointing to per-context directories, each with their own `CONTEXT.md` and `docs/adr/`.

**`CONTEXT.md` is a glossary only.** Keep it totally devoid of implementation details. Each entry: term, definition, any synonyms to avoid.

**Critical principle**: "Create files lazily — only when you have something to write."

## ADR threshold

Only propose an Architecture Decision Record when **all three** of these hold:

1. The decision is **hard to reverse**
2. It would be **surprising without context** (a future reader would wonder why)
3. It represents a **genuine trade-off** between alternatives

Skip ADRs for reversible choices, obvious choices, and ephemeral reasons ("not worth it right now").

## CONTEXT.md update format

When adding or updating a term during a session:

```markdown
## Glossary

### [Term]
[One-sentence definition. What it is, not what it does.]
Avoid: [any synonyms that would be confusing here]
```

## During grilling sessions

This skill is often run alongside `/grill-with-docs` or `/triage`. As decisions land inline:

- New term coined → add to `CONTEXT.md` immediately
- Fuzzy term sharpened → update `CONTEXT.md` right there
- Hard architectural decision made → offer an ADR
- User rejects a candidate for a load-bearing reason → offer an ADR to prevent it from being re-suggested
