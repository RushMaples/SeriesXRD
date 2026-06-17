---
name: prototype
description: Build throwaway code that answers a specific design question — logic questions get a terminal app, UI questions get togglable variations.
disable-model-invocation: true
---

# Prototype

Build throwaway code that answers a specific design question quickly. The goal is the answer, not the code.

## Core decision

**The question decides the shape.** Getting this wrong wastes the whole prototype.

- **Logic/state question** (state machines, business rules, data flow) → Interactive terminal app
- **UI/layout question** (look, feel, interaction) → Multiple UI variations togglable via URL parameters

Ask yourself: "What question am I trying to answer?" before writing anything.

## Rules

1. **Mark it clearly as temporary.** Comment at the top: `# PROTOTYPE — delete or integrate after use`.
2. **Store near the code it tests.** Don't bury it in a scratch folder three levels up.
3. **Single command to run.** Use existing project tooling — don't introduce new tools just for the prototype.
4. **In-memory state only.** No persistence unless the prototype is explicitly testing persistence.
5. **Skip polish.** No tests, no error handling, no abstractions beyond runability.
6. **Make state visible.** Print or display the full state after each action so changes are observable.
7. **Delete or integrate.** Once you have the answer, don't leave experimental code to decay.

## After the prototype

Capture the answer in durable documentation:

- Commit message: "Prototyped X, found Y. Decision: Z."
- ADR (if the finding is hard to reverse and would be surprising without context)
- Notes file referenced from the relevant issue

Then remove the prototype code from the repository.

## What makes a good prototype question

Good: "Does this state machine handle concurrent cancellations correctly?"
Good: "Which of these three layout options communicates priority better?"
Bad: "Let's build a rough version and see what happens." (no falsifiable question)

The prototype answers one question. If you find yourself answering two questions, split into two prototypes.
