# Batch Summary Schema Changelog

## Versioning Policy
- Scheme: Semantic Versioning (`MAJOR.MINOR.PATCH`)
- Compatibility baseline: `1.x`
- Major (`MAJOR`): can include breaking changes; cross-major compatibility is not guaranteed.
- Minor (`MINOR`): backward-compatible additive changes only.
- Patch (`PATCH`): non-structural fixes, clarifications, or constraints tightening that do not change required shape.

## Changelog

### 1.0.0 (2026-04-02)
- Introduced `batch_summary.json` contract with:
  - top-level summary fields (`total_archives`, `success_count`, `failure_breakdown`, etc.)
  - per-result status/error model (`success|failed|error`, `error_code`, `error_message`)
  - embedded `summary_contract` descriptions for downstream consumers.
- Added runtime JSON Schema validation in `BatchProcessor` when `jsonschema` is available.
- Added schema references in summary output:
  - `summary_schema_ref`
  - `summary_changelog_ref`
