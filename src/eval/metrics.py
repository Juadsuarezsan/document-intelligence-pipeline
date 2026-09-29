"""Evaluation metrics.

* **Field-level precision / recall / F1** per field type (micro and macro).
  A prediction counts as a true positive when the field name matches and the
  normalised values agree: amounts within 0.01, dates as ISO strings, text after
  lower-casing/whitespace/punctuation normalisation (name-like fields also
  accept containment, e.g. ``"Acme Consulting"`` vs ``"ACME CONSULTING S.A.S."``).
* **Document-level accuracy**: fraction of documents whose *critical* fields
  (see ``CRITICAL_FIELDS``) are all correct.
* **Pair F1** for FUNSD-style ``(question, answer)`` extraction.
* **Cell-level F1** for tables: positional match of normalised cell text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.extractors.normalize import normalize_amount, normalize_date, normalize_text
from src.schemas.doc_types import CRITICAL_FIELDS, DATE_FIELDS, INTEGER_FIELDS, NUMERIC_FIELDS
from src.schemas.document import DocumentType

CONTAINMENT_FIELDS = frozenset(
    {
        "vendor",
        "parties",
        "title",
        "author",
        "submitter",
        "governing_law",
        "payment_terms",
        "department",
    }
)


@dataclass
class PRF:
    """Precision/recall/F1 counters."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        """TP / (TP + FP), 0 when nothing predicted."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        """TP / (TP + FN), 0 when nothing to find."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def add(self, other: PRF) -> None:
        """Accumulate counters in place."""
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def as_dict(self) -> dict[str, float | int]:
        """Serialisable view."""
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def values_match(name: str, predicted: Any, truth: Any) -> bool:
    """Compare a predicted and a ground-truth value for field ``name``."""
    if predicted is None or truth is None:
        return False
    if name in NUMERIC_FIELDS:
        p, t = normalize_amount(predicted), normalize_amount(truth)
        return p is not None and t is not None and abs(p - t) < 0.01
    if name in INTEGER_FIELDS:
        try:
            return int(float(str(predicted))) == int(float(str(truth)))
        except ValueError:
            return False
    if name in DATE_FIELDS:
        p_date, t_date = normalize_date(str(predicted)), normalize_date(str(truth))
        return p_date is not None and p_date == t_date
    p_txt, t_txt = normalize_text(str(predicted)), normalize_text(str(truth))
    if not p_txt or not t_txt:
        return False
    if p_txt == t_txt:
        return True
    if name in CONTAINMENT_FIELDS:
        shorter, longer = sorted((p_txt, t_txt), key=len)
        return len(shorter) >= 4 and shorter in longer
    return False


def field_prf(predicted: dict[str, Any], truth: dict[str, Any]) -> dict[str, PRF]:
    """Per-field counters for one document.

    Every predicted field is either a TP (matches truth) or an FP; every truth
    field without a matching prediction is an FN.
    """
    out: dict[str, PRF] = {}
    for name, value in predicted.items():
        prf = out.setdefault(name, PRF())
        if name in truth and values_match(name, value, truth[name]):
            prf.tp += 1
        else:
            prf.fp += 1
    for name, value in truth.items():
        if name not in predicted or not values_match(name, predicted[name], value):
            out.setdefault(name, PRF()).fn += 1
    return out


def document_correct(
    predicted: dict[str, Any], truth: dict[str, Any], doc_type: DocumentType
) -> bool:
    """True when all critical fields present in the truth are predicted correctly."""
    critical = [f for f in CRITICAL_FIELDS[doc_type] if f in truth]
    return all(values_match(f, predicted.get(f), truth[f]) for f in critical)


def pair_prf(predicted: list[tuple[str, str]], truth: list[tuple[str, str]]) -> PRF:
    """Strict pair matching: both key and value must match after normalisation."""
    norm_pred = {(normalize_text(k), normalize_text(v)) for k, v in predicted}
    norm_true = {(normalize_text(k), normalize_text(v)) for k, v in truth}
    tp = len(norm_pred & norm_true)
    return PRF(tp=tp, fp=len(norm_pred) - tp, fn=len(norm_true) - tp)


def cell_prf(
    pred_headers: list[str],
    pred_rows: list[list[str]],
    true_headers: list[str],
    true_rows: list[list[str]],
) -> PRF:
    """Positional cell-level counters between a predicted and a true table."""
    pred_grid = [pred_headers] + pred_rows
    true_grid = [true_headers] + true_rows
    prf = PRF()
    for r, true_row in enumerate(true_grid):
        pred_row = pred_grid[r] if r < len(pred_grid) else []
        for c, true_cell in enumerate(true_row):
            tval = normalize_text(true_cell)
            pval = normalize_text(pred_row[c]) if c < len(pred_row) else ""
            if not tval:
                continue
            if pval == tval:
                prf.tp += 1
            else:
                prf.fn += 1
                if pval:
                    prf.fp += 1
    for r, pred_row in enumerate(pred_grid):
        true_row = true_grid[r] if r < len(true_grid) else []
        for c in range(len(true_row), len(pred_row)):
            if normalize_text(pred_row[c]):
                prf.fp += 1
    for r in range(len(true_grid), len(pred_grid)):
        prf.fp += sum(1 for cell in pred_grid[r] if normalize_text(cell))
    return prf


@dataclass
class Aggregate:
    """Accumulates metrics over many documents."""

    per_field: dict[str, PRF] = field(default_factory=dict)
    per_type: dict[str, PRF] = field(default_factory=dict)
    micro: PRF = field(default_factory=PRF)
    docs_total: int = 0
    docs_correct: int = 0
    classifier_total: int = 0
    classifier_correct: int = 0
    pairs: PRF = field(default_factory=PRF)
    pair_docs: int = 0
    latencies_ms: list[int] = field(default_factory=list)
    cost_usd: float = 0.0
    routing: dict[str, int] = field(default_factory=dict)

    def add_document(
        self,
        doc_type: str,
        per_field: dict[str, PRF],
        correct: bool | None,
        latency_ms: int,
        cost: float,
        routing: str,
        predicted_type: str | None = None,
    ) -> None:
        """Fold one document's counters into the aggregate.

        ``correct`` is ``None`` for documents without schema ground truth
        (FUNSD pairs-only cases); they still count for latency and routing.
        """
        for name, prf in per_field.items():
            self.per_field.setdefault(name, PRF()).add(prf)
            self.per_type.setdefault(doc_type, PRF()).add(prf)
            self.micro.add(prf)
        if correct is not None:
            self.docs_total += 1
            self.docs_correct += int(correct)
        self.latencies_ms.append(latency_ms)
        self.cost_usd += cost
        self.routing[routing] = self.routing.get(routing, 0) + 1
        if predicted_type is not None:
            self.classifier_total += 1
            self.classifier_correct += int(predicted_type == doc_type)

    @property
    def macro_f1(self) -> float:
        """Unweighted mean F1 over field types that appear in the truth."""
        scored = [p.f1 for p in self.per_field.values() if p.tp + p.fn > 0]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def doc_accuracy(self) -> float:
        """Fraction of documents with all critical fields correct."""
        return self.docs_correct / self.docs_total if self.docs_total else 0.0

    @property
    def classifier_accuracy(self) -> float:
        """Fraction of auto-classified documents with the right type."""
        return self.classifier_correct / self.classifier_total if self.classifier_total else 0.0

    def latency_stats(self) -> dict[str, float]:
        """Mean, p50 and p95 latency in milliseconds."""
        if not self.latencies_ms:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        lat = sorted(self.latencies_ms)
        return {
            "mean_ms": round(sum(lat) / len(lat), 1),
            "p50_ms": float(lat[len(lat) // 2]),
            "p95_ms": float(lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))]),
        }

    def as_dict(self) -> dict[str, Any]:
        """Serialisable summary."""
        return {
            "documents": len(self.latencies_ms),
            "documents_with_field_truth": self.docs_total,
            "field_f1_micro": round(self.micro.f1, 4),
            "field_precision_micro": round(self.micro.precision, 4),
            "field_recall_micro": round(self.micro.recall, 4),
            "field_f1_macro": round(self.macro_f1, 4),
            "doc_accuracy": round(self.doc_accuracy, 4),
            "classifier_accuracy": round(self.classifier_accuracy, 4),
            "classifier_documents": self.classifier_total,
            "pair_f1": round(self.pairs.f1, 4),
            "pair_precision": round(self.pairs.precision, 4),
            "pair_recall": round(self.pairs.recall, 4),
            "pair_documents": self.pair_docs,
            "latency": self.latency_stats(),
            "cost_usd_total": round(self.cost_usd, 6),
            "cost_usd_per_doc": (
                round(self.cost_usd / self.docs_total, 6) if self.docs_total else 0.0
            ),
            "routing": dict(sorted(self.routing.items())),
            "per_field": {k: v.as_dict() for k, v in sorted(self.per_field.items())},
            "per_type": {k: v.as_dict() for k, v in sorted(self.per_type.items())},
        }
