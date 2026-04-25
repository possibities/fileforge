"""Export utilities for JSON/CSV output."""

import csv
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class Exporter:
    HEADERS: Dict[str, List[str]] = {}

    @staticmethod
    def _raise_error(code: str, message: str, cause: Optional[Exception] = None) -> None:
        if cause is None:
            raise RuntimeError(f"[{code}] {message}")
        raise RuntimeError(f"[{code}] {message}") from cause

    @classmethod
    def initialize(cls, config_path: str) -> None:
        try:
            with open(config_path, "r", encoding="utf-8") as file_obj:
                headers = json.load(file_obj)
            cls.HEADERS = cls._validate_headers(headers)
        except Exception as exc:
            cls._raise_error("EXPORTER_INIT_FAILED", f"failed to load config: {exc}", cause=exc)

    @classmethod
    def get_headers(cls, template: str = "default") -> List[str]:
        if not cls.HEADERS:
            cls._raise_error("EXPORTER_NOT_INITIALIZED", "call initialize() before exporting")
        if template not in cls.HEADERS:
            cls._raise_error("EXPORTER_TEMPLATE_NOT_FOUND", f"template not found: {template}")
        return cls.HEADERS[template]

    @staticmethod
    def _validate_headers(headers: Dict) -> Dict[str, List[str]]:
        if not isinstance(headers, dict):
            raise ValueError("[INVALID_HEADERS_CONFIG] headers config must be an object")

        validated: Dict[str, List[str]] = {}
        for template, fields in headers.items():
            if not isinstance(template, str):
                raise ValueError("[INVALID_HEADERS_CONFIG] template name must be string")
            if not isinstance(fields, list) or not fields:
                raise ValueError(f"[INVALID_HEADERS_CONFIG] template '{template}' has empty fields")
            if not all(isinstance(field, str) and field.strip() for field in fields):
                raise ValueError(f"[INVALID_HEADERS_CONFIG] template '{template}' has invalid field names")
            validated[template] = fields

        return validated

    @staticmethod
    def _build_export_rows(results: List[Dict], headers: List[str]) -> List[Dict]:
        rows: List[Dict] = []
        for result in results:
            metadata = result.get("metadata", {})
            if not metadata:
                continue

            row: Dict[str, object] = {}
            for field in headers:
                value = metadata.get(field, "")
                row[field] = "" if value is None else value
            rows.append(row)
        return rows

    @classmethod
    def export_to_csv(
        cls,
        results: List[Dict],
        output_path: str,
        template: str = "default",
    ) -> int:
        if not results:
            logger.warning("[Exporter] no results to export")
            return 0

        try:
            headers = cls.get_headers(template)
            rows = cls._build_export_rows(results, headers)

            with open(output_path, "w", encoding="utf-8-sig", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)

            logger.info("[Exporter] CSV exported %s rows to %s", len(rows), output_path)
            return len(rows)
        except RuntimeError:
            raise
        except Exception as exc:
            cls._raise_error("EXPORT_CSV_FAILED", f"failed to export CSV: {exc}", cause=exc)

    @classmethod
    def export_to_json(
        cls,
        results: List[Dict],
        output_path: str,
        template: str = "default",
        indent: int = 2,
    ) -> int:
        if not results:
            logger.warning("[Exporter] no results to export")
            return 0

        try:
            headers = cls.get_headers(template)
            rows = cls._build_export_rows(results, headers)

            with open(output_path, "w", encoding="utf-8") as file_obj:
                json.dump(rows, file_obj, ensure_ascii=False, indent=indent)

            logger.info("[Exporter] JSON exported %s rows to %s", len(rows), output_path)
            return len(rows)
        except RuntimeError:
            raise
        except Exception as exc:
            cls._raise_error("EXPORT_JSON_FAILED", f"failed to export JSON: {exc}", cause=exc)
