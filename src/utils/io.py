from __future__ import annotations
from typing import Any, Dict, List
import os
import json

def ensure_dir(path: str) -> None:
    """디렉토리가 없으면 생성한다."""
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """JSON Lines 파일에 한 줄을 append 한다."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_parquet(path: str, rows: List[Dict[str, Any]]) -> None:
    """Parquet 파일로 누적 저장한다(pyarrow→fastparquet 순 Fallback)."""
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or ".")
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        table = pa.Table.from_pylist(rows)
        if os.path.exists(path):
            # append by concatenation (간단 구현)
            old = pq.read_table(path)
            table = pa.concat_tables([old, table])
        pq.write_table(table, path)
        return
    except Exception:
        pass
    try:
        import pandas as pd  # type: ignore
        engine = None
        try:
            import fastparquet  # type: ignore
            engine = "fastparquet"
        except Exception:
            engine = "pyarrow"
        df = pd.DataFrame(rows)
        if os.path.exists(path):
            df_old = pd.read_parquet(path)
            df = pd.concat([df_old, df], ignore_index=True)
        df.to_parquet(path, engine=engine)
    except Exception:
        # 마지막 Fallback: JSONL로라도 저장
        jl = os.path.splitext(path)[0] + ".jsonl"
        for r in rows:
            write_jsonl(jl, r)

