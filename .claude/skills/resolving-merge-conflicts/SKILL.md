---
name: resolving-merge-conflicts
description: Systematically resolve git merge or rebase conflicts by understanding original intent, preserving both sides where possible, and completing the process rather than aborting.
disable-model-invocation: true
---

# Resolving Merge Conflicts

A five-step approach to handling git merge/rebase conflicts that prioritizes understanding over guessing.

## Core principle

Always move forward with resolution rather than aborting the merge/rebase. Aborting just defers the conflict. Understand the intent behind each change, then reconcile.

## Step 1: Assess the current state

```bash
git status                    # see which files are conflicted
git log --oneline --graph -20 # understand the merge topology
```

Identify which files are conflicted and get a picture of what the merge is trying to accomplish.

## Step 2: Understand the context

For each conflicting change, find the primary source: **why was each change made, and what was the original intent?**

Research tools:
- `git log -p <file>` — history of changes to the conflicted file
- `git log --oneline <branch1>..<branch2>` — commits unique to each branch
- Pull request descriptions and related issues — the stated purpose of each change

Don't resolve anything until you understand both sides.

## Step 3: Resolve conflicts

**Preserve both intents where possible.** Where incompatible, pick the one matching the merge's stated goal and note the trade-off.

Rules:
- Do NOT create new behavior during resolution — you are reconciling existing intentions, not introducing new ones
- If you're uncertain which intent wins, ask before resolving
- Note any trade-offs in the commit message

## Step 4: Run automated checks

After resolving all conflicts, run the project's standard validation:

```bash
# Project-specific — run whatever applies
pytest          # tests
mypy .          # type checking
ruff check .    # linting
```

Resolve any errors introduced by the merge before completing.

## Step 5: Complete the process

```bash
git add <resolved-files>
git commit                    # for merges
# or
git rebase --continue         # for rebases
```

For rebases, continue until all commits are successfully rebased. Don't stop at the first resolution.

## What to put in the commit message

For non-trivial resolutions:

```
Merge <branch> into <branch>

Conflict in <file>: kept <what> from <branch> because <why>.
Trade-off: <what was lost from the other side>.
```
