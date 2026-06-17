---
name: ask-matt
description: Map of all the engineering skills — flows, on-ramps, and when to use each skill. Ask when you don't know which skill to reach for.
disable-model-invocation: false
---

# Ask Matt

You don't remember every skill, so ask.

A **flow** is a path through the skills. Most paths run along one **main flow**, and two **on-ramps** merge onto it. Everything else is standalone.

## The main flow: idea → ship

The route most work travels. You have an idea and want it built.

1. **`/grill-with-docs`** — sharpen the idea by interview. Start here when you **have a codebase**: it's stateful, retaining what it learns in `CONTEXT.md` and ADRs.
2. **Branch — can you settle every question in conversation?** If a question needs a runnable answer (state, business logic, a UI you have to see), detour through a prototype, bridged by `/handoff` in both directions:
   - `/handoff` out, then open a fresh session against that file
   - `/prototype` to answer the question with throwaway code
   - `/handoff` back what you learned, and reference it from the original idea thread
3. **Branch — is this a multi-session build?**
   - **Yes** → `/to-prd` (turn the thread into a PRD) → `/to-issues` (split the PRD into independently-grabbable issues). Start a fresh session per issue and kick off `/implement` by passing it the PRD and the single issue to work on.
   - **No** → `/implement` right here, in the same context window.

### Context hygiene

Keep steps 1–3 in **one unbroken context window** — don't compact or clear until after `/to-issues` — so the grilling, PRD, and issues all build on the same thinking. Each `/implement` then starts fresh, working from the issue.

The limit on this is the **smart zone**: the window (~120k tokens) within which the model still reasons sharply. If a session approaches it before `/to-issues`, `/handoff` and continue in a fresh thread.

## On-ramps

A starting situation that generates work, then merges onto the main flow.

- **Bugs and requests piling up** → `/triage`. It moves issues through triage roles and produces agent-ready issues, which `/implement` later picks up.

  Triage is only for issues **you didn't create** — bug reports, incoming feature requests, anything that arrives raw. Issues that `/to-issues` produced are already agent-ready, so **don't triage them**.

## Codebase health

Not feature work — upkeep.

- `/improve-codebase-architecture` — run whenever you have a spare moment to keep the codebase good for agents to operate in. It surfaces deepening opportunities; picking one _generates an idea_ you can take into the main flow at `/grill-with-docs`.

## Crossing sessions

- `/handoff` — when a thread is full or you need to branch off, this compacts the conversation into a markdown file. Open a new session and reference that file to carry context across. It's the bridge between context windows in either direction.
- `/compact` (built-in) — stay in the **same conversation**, letting the earlier turns be summarized. Use at **intentional breaks between phases**, when you don't mind losing the verbatim history. Don't compact mid-phase. `/handoff` forks; `/compact` continues.

## Standalone

Off the main flow entirely.

- `/grill-me` — the same relentless interview as `/grill-with-docs`, but for when you have **no codebase**. Stateless: builds no `CONTEXT.md`. Use to sharpen any plan or design that doesn't live in a repo.
- `/teach` — learn a concept over multiple sessions, using the current directory as a stateful workspace.

## Precondition

**`/setup-matt-pocock-skills`** — run before your first engineering flow to configure the issue tracker, triage labels, and doc layout the other skills assume.
