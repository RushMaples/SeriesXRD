# Correlation code catalog

`../CODE_CATALOG.json` is authoritative and assigns every retained top-level
Python file to exactly one group.

| Group | Status | Purpose |
|---|---|---|
| `latest_result_pipeline` | ACTIVE | Entrypoints that generate, assemble, audit, or validate the latest indexed results |
| `required_runtime_dependencies` | REQUIRED_DEPENDENCY | Recursive local-import closure for those entrypoints |
| `validation_support` | VALIDATION | Uniform output validator directly exercised by retained tests |
| `tests` | TEST | Regression tests for the retained algorithms |
| `workspace_maintenance` | MAINTENANCE | Read-only catalog, status, and integrity commands |

## Retention rule

A top-level Python file remains only if it is:

1. an entrypoint for a result indexed by the current frontend;
2. imported, directly or transitively, by one of those entrypoints;
3. a validator for those results or retained output contracts;
4. a direct regression test for retained code; or
5. required to verify and navigate the retained workspace.

Files stay flat because the current modules use bare sibling imports and some
formal provenance records preserve historical paths and hashes.
