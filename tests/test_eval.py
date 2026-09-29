import json
from pathlib import Path

import pytest

from src.config import Settings
from src.eval.dataset import TABLE_SET, TEST_SET, EvalCase, load_cases, load_table_cases
from src.eval.metrics import (
    PRF,
    Aggregate,
    document_correct,
    field_prf,
    pair_prf,
    values_match,
)
from src.eval.pipelines import PIPELINE_LABELS, build_pipeline
from src.eval.report import README_END, README_START, mandatory_table, render_results, update_readme
from src.eval.run_eval import evaluate_tables, main


def test_values_match_rules() -> None:
    assert values_match("total", "1,200.00", 1200)
    assert not values_match("total", "1,200.10", 1200)
    assert values_match("issue_date", "March 3, 2026", "2026-03-03")
    assert values_match("term_months", "24", 24.0)
    assert values_match("vendor", "ACME Consulting", "Acme Consulting S.A.S.")
    assert not values_match("invoice_number", "INV-1", "INV-10")
    assert not values_match("total", None, 1)


def test_field_prf_and_document_correct() -> None:
    truth = {"invoice_number": "A-1", "total": 10.0, "issue_date": "2026-01-01"}
    pred = {"invoice_number": "A-1", "total": 11.0, "vendor": "X"}
    prf = field_prf(pred, truth)
    assert prf["invoice_number"].tp == 1
    assert prf["total"].fp == 1 and prf["total"].fn == 1
    assert prf["vendor"].fp == 1 and prf["issue_date"].fn == 1
    assert not document_correct(pred, truth, "invoice")
    assert document_correct(
        {"invoice_number": "A-1", "total": 10, "issue_date": "2026-01-01"}, truth, "invoice"
    )


def test_pair_prf_and_aggregate() -> None:
    prf = pair_prf([("Date", "9/3/92"), ("To", "Bob")], [("date", "9/3/92"), ("From", "Al")])
    assert (prf.tp, prf.fp, prf.fn) == (1, 1, 1)
    agg = Aggregate()
    agg.add_document("invoice", {"total": PRF(tp=1)}, True, 10, 0.0, "auto_approve", "invoice")
    agg.add_document("invoice", {"total": PRF(fn=1)}, False, 30, 0.0, "human_review", "form")
    agg.add_document("form", {}, None, 5, 0.0, "human_review", None)
    d = agg.as_dict()
    assert d["documents"] == 3 and d["documents_with_field_truth"] == 2
    assert d["doc_accuracy"] == 0.5 and d["classifier_accuracy"] == 0.5
    assert d["field_f1_micro"] == pytest.approx(2 / 3, abs=1e-3)
    assert d["latency"]["p95_ms"] == 30.0


def test_dataset_is_loaded_and_marked() -> None:
    cases = load_cases(TEST_SET)
    assert len(cases) >= 100
    assert {c.source for c in cases} == {"synthetic", "funsd"}
    assert all(c.field_truth or c.pair_truth for c in cases)
    assert len(load_table_cases(TABLE_SET)) == 30


def test_pipeline_labels_and_key_requirement(settings: Settings) -> None:
    assert set(PIPELINE_LABELS) == {"heuristic", "fallback", "vision"}
    with pytest.raises(RuntimeError):
        build_pipeline("vision", settings)
    assert build_pipeline("heuristic", settings) is not None


def test_evaluate_tables_scores_synthetic_pdfs() -> None:
    result = evaluate_tables(load_table_cases()[:4])
    assert result["documents"] == 4 and result["detected"] >= 3
    assert 0.0 < result["f1"] <= 1.0


def test_run_eval_smoke_writes_run_and_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.eval import run_eval

    monkeypatch.setattr(run_eval, "RESULTS_MD", tmp_path / "RESULTS.md")
    monkeypatch.setattr(run_eval, "README", tmp_path / "README.md")
    (tmp_path / "README.md").write_text(
        f"# x\n{README_START}\nold\n{README_END}\n", encoding="utf-8"
    )
    rc = main(
        [
            "--limit",
            "3",
            "--skip-scanned",
            "--skip-tables",
            "--out-dir",
            str(tmp_path),
            "--update-readme",
            "--name",
            "smoke",
        ]
    )
    assert rc == 0
    run_file = next(tmp_path.glob("*-smoke.json"))
    run = json.loads(run_file.read_text(encoding="utf-8"))
    assert run["pipelines"]["heuristic"]["summary"]["documents"] == 3
    assert run["pipelines"]["fallback"]["summary"] is None
    results = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "pendiente (requiere ANTHROPIC_API_KEY)" in results
    assert "fallback determinista, sin LLM" in results
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "old" not in readme and "| Pipeline | Field F1 |" in readme
    assert mandatory_table(run).count("\n") == 4
    assert "Tabla obligatoria" in render_results(run)


def test_update_readme_without_markers_returns_false(tmp_path: Path) -> None:
    p = tmp_path / "README.md"
    p.write_text("no markers", encoding="utf-8")
    assert (
        update_readme(p, {"run_file": "x", "generated_at": "y", "pipelines": {}, "tables": {}})
        is False
    )
    assert update_readme(tmp_path / "missing.md", {}) is False


def test_eval_case_properties() -> None:
    case = EvalCase(
        id="c", source="funsd", doc_type="form", text="t", ground_truth={"pairs": [["a", "b"]]}
    )
    assert case.field_truth == {} and case.pair_truth == [("a", "b")]
