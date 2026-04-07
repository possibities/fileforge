#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate batch summary JSON files against schema and semver policy."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover - optional dependency in runtime
    Draft202012Validator = None


class BatchSummaryValidationError(RuntimeError):
    """Raised when a batch summary payload fails semantic/schema validation."""


def parse_semver(version: str) -> Optional[Tuple[int, int, int]]:
    if not isinstance(version, str):
        return None
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def is_supported_major(version: str, expected_major: int) -> bool:
    parsed = parse_semver(version)
    if parsed is None:
        return False
    major, _minor, _patch = parsed
    return major == expected_major


def _load_json_file(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BatchSummaryValidationError(
            f"Failed to load JSON file: {path} ({exc})"
        ) from exc


def validate_summary_payload(
    summary_payload: Dict,
    schema_payload: Dict,
    expected_major: int = 1,
) -> None:
    version = summary_payload.get("summary_schema_version")
    if not is_supported_major(version, expected_major):
        raise BatchSummaryValidationError(
            f"Unsupported summary schema major version: {version!r}. "
            f"Expected major {expected_major}."
        )

    if Draft202012Validator is None:
        raise BatchSummaryValidationError(
            "jsonschema is not installed; cannot validate summary payload."
        )

    validator = Draft202012Validator(schema_payload)
    errors = sorted(validator.iter_errors(summary_payload), key=lambda err: list(err.path))
    if not errors:
        return

    first = errors[0]
    location = "/".join(str(item) for item in first.path) or "<root>"
    raise BatchSummaryValidationError(
        f"Summary does not match schema at '{location}': {first.message}"
    )


def validate_summary_file(
    summary_path: Union[str, Path],
    schema_path: Union[str, Path] = "config/batch_summary.schema.json",
    expected_major: int = 1,
) -> None:
    summary_payload = _load_json_file(Path(summary_path))
    schema_payload = _load_json_file(Path(schema_path))
    validate_summary_payload(summary_payload, schema_payload, expected_major=expected_major)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate batch summary JSON against schema.")
    parser.add_argument("summary", help="Path to batch_summary.json")
    parser.add_argument(
        "--schema",
        default="config/batch_summary.schema.json",
        help="Path to schema file (default: config/batch_summary.schema.json)",
    )
    parser.add_argument(
        "--expected-major",
        type=int,
        default=1,
        help="Expected schema major version (default: 1)",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        validate_summary_file(
            summary_path=args.summary,
            schema_path=args.schema,
            expected_major=args.expected_major,
        )
    except BatchSummaryValidationError as exc:
        print(f"[INVALID] {exc}")
        return 1

    print("[OK] Summary is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
