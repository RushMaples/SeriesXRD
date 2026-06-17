# Issue tracker

Issues for this repository are tracked in **GitHub Issues** on `rmaples3/BulkXRD`.

## How skills interact with it

Skills that read from or write to the issue tracker (`to-issues`, `triage`,
`to-prd`, `qa`, and similar) should use the GitHub CLI (`gh`):

- **Create an issue:** `gh issue create --title "..." --body "..." --label "..."`
- **List issues:** `gh issue list --label "..."`
- **View an issue:** `gh issue view <number>`
- **Comment:** `gh issue comment <number> --body "..."`
- **Apply / remove labels:** `gh issue edit <number> --add-label "..." --remove-label "..."`
- **Close:** `gh issue close <number>`

> In remote execution environments the `gh` CLI may be unavailable; use the
> GitHub MCP tools (`mcp__github__*`) as the equivalent — e.g.
> `issue_write` to create/update, `list_issues` / `issue_read` to read,
> `add_issue_comment` to comment.

## Labels

Triage label strings live in `docs/agents/triage-labels.md`. Apply those exact
strings so the skill matches existing labels instead of creating duplicates.
