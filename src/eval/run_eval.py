"""Evaluation entry point: ``python -m eval.run``.

Runs the heuristic pipeline over the evaluation set in three input modes
(text, text-layer PDF, scanned PDF via OCR), the table extractor over the
synthetic tables, and — only when an API key is configured — the two Claude
pipelines. Writes ``eval/runs/<date>-<name>.json`` and regenerates
``eval/RESULTS.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import Settings, get_settings
from src.eval.dataset import ROOT, EvalCase, TableCase, load_cases, load_table_cases
from src.eval.metrics import Aggregate, cell_prf, document_correct, field_prf, pair_prf
from src.eval.pipelines import (
    PIPELINE_LABELS,
    CaseOutcome,
    InputMode,
    PipelineName,
    build_pipeline,
    run_case,
)
from src.eval.report import render_results, update_readme
from src.extractors.heuristic import HeuristicExtractor
from src.parsers.ocr_parser import tesseract_available
from src.synth.documents import SEED
from src.synth.pdf import render_table_pdf
from src.tables.extractor import extract_tables_pdfplumber

RUNS_DIR = ROOT / "eval" / "runs"
RESULTS_MD = ROOT / "eval" / "RESULTS.md"
README = ROOT / "README.md"


def git_commit() -> str:
    """Short hash of HEAD (``unknown`` outside a checkout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.stdout.strip()


def evaluate_outcomes(
    cases: list[EvalCase], outcomes: list[CaseOutcome]
) -> tuple[Aggregate, list[dict[str, Any]]]:
    """Score outcomes against ground truth; also return per-case details."""
    agg = Aggregate()
    by_id = {c.id: c for c in cases}
    details: list[dict[str, Any]] = []
    for out in outcomes:
        case = by_id[out.case_id]
        truth = case.field_truth
        prf = field_prf(out.predicted, truth) if truth else {}
        correct = document_correct(out.predicted, truth, case.doc_type) if truth else None
        agg.add_document(
            case.doc_type,
            prf,
            correct,
            out.latency_ms,
            out.cost_usd,
            out.routing,
            out.predicted_type,
        )
        if case.pair_truth and case.source == "funsd":
            pp = pair_prf(out.pairs, case.pair_truth)
            agg.pairs.add(pp)
            agg.pair_docs += 1
        total_tp = sum(p.tp for p in prf.values())
        total = sum(p.tp + p.fp + p.fn for p in prf.values())
        doc_f1 = (2 * total_tp / (total + total_tp)) if total + total_tp else 0.0
        details.append(
            {
                "case_id": case.id,
                "doc_type": case.doc_type,
                "source": case.source,
                "mode": out.mode,
                "f1": round(doc_f1, 4),
                "correct": correct,
                "predicted_type": out.predicted_type,
                "routing": out.routing,
                "confidence": out.confidence,
                "latency_ms": out.latency_ms,
                "cost_usd": out.cost_usd,
                "missed": sorted(k for k, p in prf.items() if p.fn),
                "wrong": sorted(k for k, p in prf.items() if p.fp),
                "findings": out.findings,
                "predicted": out.predicted,
            }
        )
    return agg, details


def evaluate_tables(table_cases: list[TableCase]) -> dict[str, Any]:
    """Render each table to PDF, extract it back and score cell-level F1."""
    from src.eval.metrics import PRF

    total, ruled, unruled = PRF(), PRF(), PRF()
    detected = 0
    latencies: list[int] = []
    per_table: list[dict[str, Any]] = []
    for tc in table_cases:
        pdf = render_table_pdf(
            tc.headers, tc.rows, ruled=tc.ruled, intro=f"Table {tc.id} ({tc.kind})"
        )
        start = time.perf_counter()
        found = extract_tables_pdfplumber(pdf)
        latencies.append(int((time.perf_counter() - start) * 1000))
        if found:
            detected += 1
            best = max(found, key=lambda t: cell_prf(t.headers, t.rows, tc.headers, tc.rows).f1)
            prf = cell_prf(best.headers, best.rows, tc.headers, tc.rows)
        else:
            prf = cell_prf([], [], tc.headers, tc.rows)
        total.add(prf)
        (ruled if tc.ruled else unruled).add(prf)
        per_table.append({"id": tc.id, "kind": tc.kind, "ruled": tc.ruled, **prf.as_dict()})
    lat = sorted(latencies)
    return {
        "documents": len(table_cases),
        "ruled": sum(1 for t in table_cases if t.ruled),
        "unruled": sum(1 for t in table_cases if not t.ruled),
        "detected": detected,
        "precision": round(total.precision, 4),
        "recall": round(total.recall, 4),
        "f1": round(total.f1, 4),
        "f1_ruled": round(ruled.f1, 4),
        "f1_unruled": round(unruled.f1, 4),
        "latency": {
            "mean_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
            "p95_ms": float(lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))]) if lat else 0.0,
        },
        "backend": "pdfplumber",
        "per_table": per_table,
    }


async def run_pipeline(
    name: PipelineName, cases: list[EvalCase], modes: list[InputMode], settings: Settings
) -> dict[str, Any]:
    """Run one pipeline over all cases in every requested mode."""
    pipeline = build_pipeline(name, settings)
    heuristic = HeuristicExtractor()
    result: dict[str, Any] = {"label": PIPELINE_LABELS[name], "modes": {}, "cases": {}}
    for mode in modes:
        outcomes: list[CaseOutcome] = []
        for i, case in enumerate(cases, start=1):
            outcomes.append(await run_case(pipeline, name, case, mode, heuristic, settings))
            if i % 25 == 0:
                logger.info("{} mode={} {}/{}", name, mode, i, len(cases))
        agg, details = evaluate_outcomes(cases, outcomes)
        result["modes"][mode] = agg.as_dict()
        result["cases"][mode] = details
    # Headline numbers come from real PDFs through pdfplumber, not from pre-extracted text.
    headline = "pdf" if "pdf" in result["modes"] else modes[0]
    result["headline_mode"] = headline
    result["summary"] = result["modes"][headline]
    scored = [d for d in result["cases"][headline] if d["source"] == "synthetic"]
    result["worst_cases"] = sorted(scored, key=lambda d: (d["f1"], d["confidence"]))[:10]
    return result


def build_run(
    name: str,
    cases: list[EvalCase],
    table_cases: list[TableCase],
    pipelines: dict[str, Any],
    tables: dict[str, Any] | None,
    settings: Settings,
) -> dict[str, Any]:
    """Assemble the run dictionary that is saved to ``eval/runs``."""
    stamp = datetime.now(UTC)
    return {
        "name": name,
        "generated_at": stamp.isoformat(timespec="seconds"),
        "run_file": f"eval/runs/{stamp.date().isoformat()}-{name}.json",
        "git_commit": git_commit(),
        "model": settings.anthropic_model,
        "llm_enabled": settings.llm_enabled,
        "tesseract_available": tesseract_available(settings.tesseract_cmd),
        "dataset": {
            "documents": len(cases),
            "synthetic": sum(1 for c in cases if c.source == "synthetic"),
            "funsd": sum(1 for c in cases if c.source == "funsd"),
            "tables": len(table_cases),
            "seed": SEED,
        },
        "pipelines": pipelines,
        "tables": tables,
    }


async def run_all(args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    """Execute the evaluation according to CLI arguments and return the run."""
    cases = load_cases()
    table_cases = load_table_cases()
    if args.limit:
        cases = cases[: args.limit]
        table_cases = table_cases[: args.limit]
    modes: list[InputMode] = ["text", "pdf"]
    if not args.skip_scanned and tesseract_available(settings.tesseract_cmd):
        modes.append("scanned")
    elif not args.skip_scanned:
        logger.warning("tesseract not found; skipping the scanned mode")
    pipelines: dict[str, Any] = {}
    for name in args.pipelines:
        if name != "heuristic" and not settings.llm_enabled:
            logger.warning("pipeline {} skipped: ANTHROPIC_API_KEY not set", name)
            pipelines[name] = {
                "label": PIPELINE_LABELS[name],
                "summary": None,
                "status": "pendiente (requiere ANTHROPIC_API_KEY)",
            }
            continue
        pipeline_modes = modes if name == "heuristic" else ["pdf"]
        pipelines[name] = await run_pipeline(name, cases, pipeline_modes, settings)
    tables = None if args.skip_tables else evaluate_tables(table_cases)
    return build_run(args.name, cases, table_cases, pipelines, tables, settings)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the document-intelligence evaluation")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["heuristic", "fallback", "vision"],
        choices=list(PIPELINE_LABELS),
    )
    parser.add_argument("--name", default="heuristic-baseline")
    parser.add_argument("--limit", type=int, default=0, help="only the first N cases (smoke run)")
    parser.add_argument("--skip-scanned", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument(
        "--no-write", action="store_true", help="do not write eval/runs or RESULTS.md"
    )
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args(argv)
    settings = get_settings()
    run = asyncio.run(run_all(args, settings))
    summary = run["pipelines"].get("heuristic", {}).get("summary")
    if summary:
        logger.info(
            "heuristic field_f1={} doc_acc={} table_f1={}",
            summary["field_f1_micro"],
            summary["doc_accuracy"],
            (run.get("tables") or {}).get("f1"),
        )
    if args.no_write:
        print(json.dumps({k: v for k, v in run.items() if k != "pipelines"}, indent=2, default=str))
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_path = (
        ROOT / run["run_file"]
        if args.out_dir == RUNS_DIR
        else args.out_dir / Path(run["run_file"]).name
    )
    run_path.write_text(
        json.dumps(run, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    RESULTS_MD.write_text(render_results(run), encoding="utf-8")
    logger.info("wrote {} and {}", run_path, RESULTS_MD)
    if args.update_readme and update_readme(README, run):
        logger.info("README metrics block updated")
    return 0
