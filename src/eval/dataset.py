"""Evaluation set I/O (``data/eval/test_set.jsonl`` and ``data/eval/tables.jsonl``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.schemas.document import DocumentType

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "eval"
TEST_SET = DATA_DIR / "test_set.jsonl"
TABLE_SET = DATA_DIR / "tables.jsonl"


class TableSpec(BaseModel):
    """Ground-truth table (header + body) for cell-level F1."""

    headers: list[str]
    rows: list[list[str]]
    ruled: bool = True


class EvalCase(BaseModel):
    """One document of the evaluation set."""

    id: str
    source: str = Field(description="``synthetic`` or ``funsd``")
    doc_type: DocumentType
    template: str = ""
    text: str
    ground_truth: dict[str, Any]
    table: TableSpec | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def field_truth(self) -> dict[str, Any]:
        """Ground-truth schema fields (excludes the free ``pairs`` list)."""
        return {k: v for k, v in self.ground_truth.items() if k != "pairs"}

    @property
    def pair_truth(self) -> list[tuple[str, str]]:
        """Ground-truth key/value pairs (FUNSD-style)."""
        return [(str(p[0]), str(p[1])) for p in self.ground_truth.get("pairs", [])]


class TableCase(BaseModel):
    """One table document for the table-extraction evaluation."""

    id: str
    kind: str
    headers: list[str]
    rows: list[list[str]]
    ruled: bool = True


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records one JSON object per line (sorted keys for stable hashes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def load_cases(path: Path = TEST_SET) -> list[EvalCase]:
    """Load the document evaluation set."""
    return [EvalCase.model_validate(r) for r in _read_jsonl(path)]


def load_table_cases(path: Path = TABLE_SET) -> list[TableCase]:
    """Load the table evaluation set."""
    return [TableCase.model_validate(r) for r in _read_jsonl(path)]
