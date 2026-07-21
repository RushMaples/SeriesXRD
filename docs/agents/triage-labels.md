# Triage labels

> Development-automation configuration (see "Development transparency" in
> the repository's `GOVERNANCE.md`) — not user documentation.

The `triage` skill moves an incoming issue through a state machine and applies
one of the labels below. These are the **canonical default strings** — they
equal their role names. If you rename a label in GitHub, update the right-hand
value here so the skill applies the string that actually exists.

| Role                                          | Label string      |
| --------------------------------------------- | ----------------- |
| Maintainer needs to evaluate                  | `needs-triage`    |
| Waiting on reporter                           | `needs-info`      |
| Fully specified, AFK-ready (agent can pick up)| `ready-for-agent` |
| Needs human implementation                    | `ready-for-human` |
| Will not be actioned                          | `wontfix`         |

These labels may need to be created in the GitHub repo before first use:

```sh
gh label create needs-triage    --description "Maintainer needs to evaluate"
gh label create needs-info      --description "Waiting on reporter"
gh label create ready-for-agent --description "Fully specified, AFK-ready"
gh label create ready-for-human --description "Needs human implementation"
gh label create wontfix         --description "Will not be actioned"
```
