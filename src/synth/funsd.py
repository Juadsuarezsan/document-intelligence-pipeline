"""FUNSD annotation loader: reconstruct reading-order text and question/answer pairs.

FUNSD (Jaume et al., 2019) annotates scanned forms with entities labelled
``question``/``answer``/``header``/``other`` and ``linking`` edges. The
linked ``question -> answer`` pairs are the ground truth for key/value
extraction; the text is rebuilt from entity boxes so an OCR-like reading
order is available without the page images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LINE_TOLERANCE = 12  # pixels; entities whose vertical centres are this close share a line


@dataclass
class FunsdDoc:
    """One FUNSD form with reconstructed text and its QA pairs."""

    id: str
    text: str
    pairs: list[tuple[str, str]]
    n_entities: int
    labels: dict[str, int] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Serialise for ``test_set.jsonl``."""
        return {
            "id": f"funsd-{self.id}",
            "source": "funsd",
            "doc_type": "form",
            "template": "funsd_annotation",
            "text": self.text,
            "ground_truth": {"pairs": [list(p) for p in self.pairs]},
            "table": None,
            "notes": [
                f"entities={self.n_entities}",
                "labels=" + json.dumps(self.labels, sort_keys=True),
            ],
        }


def _reading_order_text(entities: list[dict[str, Any]]) -> str:
    items = []
    for e in entities:
        text = str(e.get("text", "")).strip()
        if not text:
            continue
        x0, y0, x1, y1 = e["box"]
        items.append(((y0 + y1) / 2.0, x0, text))
    items.sort()
    lines: list[list[tuple[float, str]]] = []
    current_y: float | None = None
    for cy, x0, text in items:
        if current_y is None or abs(cy - current_y) > LINE_TOLERANCE:
            lines.append([])
            current_y = cy
        lines[-1].append((x0, text))
    return "\n".join("  ".join(t for _, t in sorted(line)) for line in lines)


def load_funsd_annotation(path: Path) -> FunsdDoc:
    """Parse one FUNSD ``*.json`` annotation file.

    Args:
        path: Path to the annotation.

    Returns:
        The document with text in reading order and ``(question, answer)`` pairs.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    entities: list[dict[str, Any]] = data["form"]
    by_id = {e["id"]: e for e in entities}
    pairs: list[tuple[str, str]] = []
    labels: dict[str, int] = {}
    for e in entities:
        labels[e["label"]] = labels.get(e["label"], 0) + 1
        if e["label"] != "question":
            continue
        for src, dst in e.get("linking", []):
            if src != e["id"]:
                continue
            target = by_id.get(dst)
            if target and target["label"] == "answer" and str(target["text"]).strip():
                pairs.append((str(e["text"]).strip(), str(target["text"]).strip()))
    return FunsdDoc(
        id=path.stem,
        text=_reading_order_text(entities),
        pairs=pairs,
        n_entities=len(entities),
        labels=labels,
    )


def load_funsd_dir(directory: Path) -> list[FunsdDoc]:
    """Load every annotation in a directory, sorted by file name."""
    return [load_funsd_annotation(p) for p in sorted(directory.glob("*.json"))]
