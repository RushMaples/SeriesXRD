# Triage labels

Canonical role → actual label string. This repo uses the canonical names verbatim.

## Category (exactly one per issue)

| Role          | Label string  |
| ------------- | ------------- |
| bug           | `bug`         |
| enhancement   | `enhancement` |

## State (exactly one per triaged issue)

| Role             | Label string       | Meaning                                  |
| ---------------- | ------------------ | ---------------------------------------- |
| needs-triage     | `needs-triage`     | Maintainer needs to evaluate             |
| needs-info       | `needs-info`       | Waiting on reporter                      |
| ready-for-agent  | `ready-for-agent`  | Fully specified, AFK-ready for an agent  |
| ready-for-human  | `ready-for-human`  | Needs human implementation               |
| wontfix          | `wontfix`          | Will not be actioned                     |

Labels are created in the repo on first use if they don't exist yet.
