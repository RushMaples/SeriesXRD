# Domain docs

> Development-automation configuration (see "Development transparency" in
> the repository's `GOVERNANCE.md`) — not user documentation.

This repository is **single-context**: one domain context covering the whole
`seriesxrd` package.

## Layout

- `CONTEXT.md` (repo root) — the project's domain language and model.
- `docs/adr/` (repo root) — Architecture Decision Records.

> Neither `CONTEXT.md` nor `docs/adr/` exists yet. The `domain-modeling` skill
> creates and maintains `CONTEXT.md`; ADRs are added under `docs/adr/` as
> architectural decisions land.

## Consumer rules

Skills that read domain docs (`improve-codebase-architecture`,
`diagnosing-bugs`, `tdd`, `domain-modeling`):

- Read `CONTEXT.md` at the repo root to learn the project's domain vocabulary
  before reasoning about behaviour.
- Read `docs/adr/` for prior architectural decisions before proposing changes
  that touch architecture; don't relitigate a decision already recorded there.
- This is a single-context repo, so there is **no** `CONTEXT-MAP.md` — do not
  look for per-context `CONTEXT.md` files in subdirectories.

For high-level package structure and existing design decisions not yet captured
as ADRs, see `CLAUDE.md` at the repo root.
