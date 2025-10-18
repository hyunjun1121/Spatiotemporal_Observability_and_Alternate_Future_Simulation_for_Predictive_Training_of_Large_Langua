from __future__ import annotations
from typing import Any, Dict
import json
import os

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    from pydantic import BaseModel, ValidationError
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore
    ValidationError = Exception  # type: ignore


def load_yaml(path: str) -> Dict[str, Any]:
    """YAML 파일을 로드하여 dict로 반환한다.(없으면 빈 dict)"""
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: str) -> Dict[str, Any]:
    """JSON 파일을 로드하여 dict로 반환한다.(없으면 빈 dict)"""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_log_record(record: Dict[str, Any], schema_path: str) -> None:
    """로그 레코드가 schema(JSON Schema 간소형)에 부합하는지 최소 검증한다."""
    schema = load_json(schema_path)
    required = schema.get("required", [])
    props = schema.get("properties", {})

    for k in required:
        if k not in record:
            raise ValueError(f"Missing required log field: {k}")

    for k, v in list(record.items()):
        if k in props:
            typ = props[k].get("type")
            if typ == "number" and not isinstance(v, (int, float)):
                raise ValueError(f"Field {k} must be number, got {type(v)}")
            if typ == "integer" and not isinstance(v, int):
                raise ValueError(f"Field {k} must be integer, got {type(v)}")
            if typ == "boolean" and not isinstance(v, bool):
                raise ValueError(f"Field {k} must be boolean, got {type(v)}")

