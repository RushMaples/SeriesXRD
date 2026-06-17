# Domain docs

This repo is **single-context**.

## Layout

- `CONTEXT.md` (repo root) — domain glossary ONLY. No implementation details.
  Each entry: term, one-sentence definition, synonyms to avoid. Created lazily —
  only when there's something to write.
- `docs/adr/` (repo root) — Architecture Decision Records. One file per decision.

## Consumer rules

Skills that read these: `improve-codebase-architecture`, `diagnosing-bugs`, `tdd`,
`to-issues`, `to-prd`, `domain-modeling`.

- Read `CONTEXT.md` before naming things, so vocabulary matches the project's
  domain language.
- Respect ADRs in `docs/adr/` — don't re-litigate decisions already recorded.
- Add an ADR only when a decision is hard to reverse, surprising without context,
  AND a genuine trade-off between alternatives.

## Note

`CLAUDE.md` already has a "Key design decisions (don't relitigate)" section that
functions as an informal ADR log. Formal ADRs under `docs/adr/` can be migrated
from it lazily via the `domain-modeling` skill as decisions get revisited.
