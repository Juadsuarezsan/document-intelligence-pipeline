"""Render ``eval/RESULTS.md`` (and the README metrics block) from a saved run."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.eval.pipelines import PIPELINE_LABELS, PipelineName

PENDING = "pendiente (requiere ANTHROPIC_API_KEY)"
README_START = "<!-- METRICS:START -->"
README_END = "<!-- METRICS:END -->"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f} %"


def _row(label: str, summary: dict[str, Any] | None, table_f1: float | None) -> str:
    if summary is None:
        return f"| {label} | {PENDING} | {PENDING} | {PENDING} | {PENDING} | {PENDING} |"
    lat = summary["latency"]
    return (
        f"| {label} | {_pct(summary['field_f1_micro'])} | {_pct(summary['doc_accuracy'])} | "
        f"{_pct(table_f1)} | {lat['mean_ms']:.0f} ms (p95 {lat['p95_ms']:.0f} ms) | "
        f"${summary['cost_usd_per_doc']:.4f} |"
    )


def mandatory_table(run: dict[str, Any]) -> str:
    """The spec's 3x5 comparison table in Markdown."""
    lines = [
        "| Pipeline | Field F1 | Doc Accuracy | Table F1 | Latencia | Costo/doc |",
        "|---|---|---|---|---|---|",
    ]
    pipelines = run["pipelines"]
    tables = run.get("tables", {})
    name: PipelineName
    for name in ("heuristic", "fallback", "vision"):
        entry = pipelines.get(name)
        summary = entry.get("summary") if entry else None
        table_f1 = (
            tables.get("f1")
            if (name == "heuristic" and tables)
            else (tables.get("f1") if summary and tables else None)
        )
        lines.append(_row(PIPELINE_LABELS[name], summary, table_f1))
    return "\n".join(lines)


def _prf_table(title: str, rows: dict[str, dict[str, float | int]]) -> str:
    out = [
        f"### {title}",
        "",
        "| Campo | P | R | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in rows.items():
        out.append(
            f"| `{name}` | {_pct(float(m['precision']))} | {_pct(float(m['recall']))} | {_pct(float(m['f1']))} | {m['tp']} | {m['fp']} | {m['fn']} |"
        )
    return "\n".join(out)


def render_results(run: dict[str, Any]) -> str:
    """Full Markdown report for a run dictionary (see ``run_eval.build_run``)."""
    heur = run["pipelines"].get("heuristic", {})
    modes = heur.get("modes", {})
    tables = run.get("tables")
    parts: list[str] = [
        "# Resultados de evaluación",
        "",
        f"Generado por `python -m eval.run` el {run['generated_at']} · corrida `{run['run_file']}` · "
        f"commit `{run['git_commit']}`.",
        "",
        "> **Lectura obligatoria.** Solo la fila *Unstructured only* fue medida en esta corrida: es el "
        "**fallback determinista, sin LLM** (pdfplumber/OCR + reglas). No representa la calidad del "
        "sistema completo. Las filas con Claude Vision requieren `ANTHROPIC_API_KEY` y quedan "
        "pendientes; el código que las produce está implementado y probado con mocks.",
        "",
        f"Dataset: {run['dataset']['documents']} documentos "
        f"({run['dataset']['synthetic']} sintéticos generados con semilla {run['dataset']['seed']} + "
        f"{run['dataset']['funsd']} formularios FUNSD reales de `data/funsd_sample/`), "
        f"{run['dataset']['tables']} tablas sintéticas en PDF. Los datos sintéticos están marcados como tales "
        "en `data/eval/test_set.jsonl` (`source`).",
        "",
        "## Tabla obligatoria",
        "",
        mandatory_table(run),
        "",
        "Field F1 es micro-F1 sobre pares (campo, valor) contra la verdad de terreno; Doc Accuracy exige "
        "todos los campos críticos correctos; Table F1 es F1 a nivel de celda (posicional) sobre las tablas "
        "sintéticas; latencia y costo se miden por documento. La fila heurística usa el modo `pdf` (PDF real vía pdfplumber); "
        "el modo `text` (texto ya extraído) es el techo de las reglas y se muestra aparte.",
        "",
    ]
    if modes:
        parts += [
            "## Fila heurística por modo de entrada",
            "",
            "| Modo | Docs | Field F1 (micro) | Field F1 (macro) | Doc Accuracy | Clasificador | Latencia media | p95 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for mode, s in modes.items():
            lat = s["latency"]
            parts.append(
                f"| `{mode}` | {s['documents']} | {_pct(s['field_f1_micro'])} | {_pct(s['field_f1_macro'])} | {_pct(s['doc_accuracy'])} | {_pct(s['classifier_accuracy'])} ({s['classifier_documents']}) | {lat['mean_ms']:.0f} ms | {lat['p95_ms']:.0f} ms |"
            )
        parts.append("")
        if "text" in modes:
            text_mode = modes["text"]
            parts += [
                _prf_table("Field F1 por tipo de documento (modo text)", text_mode["per_type"]),
                "",
                _prf_table("Field F1 por tipo de campo (modo text)", text_mode["per_field"]),
                "",
            ]
            parts += [
                "### FUNSD: pares pregunta/respuesta (modo text)",
                "",
                f"Pares (question → answer) enlazados en las 5 anotaciones: F1 estricto {_pct(text_mode['pair_f1'])} "
                f"(P {_pct(text_mode['pair_precision'])}, R {_pct(text_mode['pair_recall'])}) sobre {text_mode['pair_documents']} documentos. "
                "El baseline solo reconoce pares `clave: valor` en la misma línea del texto reconstruido; la "
                "mayoría de las respuestas FUNSD están en otra línea o en casillas, de ahí el recall bajo.",
                "",
                "### Distribución de routing (modo text)",
                "",
                "| Routing | Docs |",
                "|---|---|",
            ]
            parts += [f"| `{k}` | {v} |" for k, v in text_mode["routing"].items()]
            parts.append("")
    if tables:
        parts += [
            "## Extracción de tablas (pdfplumber)",
            "",
            f"{tables['documents']} tablas sintéticas ({tables['ruled']} con líneas, {tables['unruled']} sin líneas). "
            f"Cell F1 global {_pct(tables['f1'])} (P {_pct(tables['precision'])}, R {_pct(tables['recall'])}); "
            f"con líneas {_pct(tables['f1_ruled'])}, sin líneas {_pct(tables['f1_unruled'])}. "
            f"Tablas detectadas: {tables['detected']}/{tables['documents']}. Latencia media {tables['latency']['mean_ms']:.0f} ms.",
            "",
        ]
    worst = heur.get("worst_cases", [])
    if worst:
        parts += [
            "## Diez peores casos (fila heurística, modo pdf)",
            "",
            "| # | Caso | Tipo | F1 | Campos fallidos | Hallazgos |",
            "|---|---|---|---|---|---|",
        ]
        for i, w in enumerate(worst, start=1):
            parts.append(
                f"| {i} | `{w['case_id']}` | {w['doc_type']} | {_pct(w['f1'])} | {', '.join(w['missed']) or '—'} | {', '.join(w['findings']) or '—'} |"
            )
        parts += ["", "El análisis de causas está en `docs/error_analysis.md`.", ""]
    parts += [
        "## Pendiente (requiere llave o recursos externos)",
        "",
        "- Filas *Unstructured + Claude Vision fallback* y *Claude Vision directo*: `ANTHROPIC_API_KEY`.",
        "- LLM-as-judge (rúbrica numerada en `src/validators/llm_judge.py`): `ANTHROPIC_API_KEY`.",
        "- FUNSD completo (199 formularios) y FinTabNet: descarga con `scripts/download_data.py` en una red sin restricciones.",
        "- Document-level accuracy sobre DocVQA: cuenta en el portal RRC.",
        "",
    ]
    return "\n".join(parts)


def readme_block(run: dict[str, Any]) -> str:
    """Markdown snippet inserted between the README metric markers."""
    return "\n".join(
        [
            README_START,
            f"Última corrida: `{run['run_file']}` ({run['generated_at']}). Solo la fila heurística está medida "
            "(fallback determinista, sin LLM); las de Claude Vision requieren `ANTHROPIC_API_KEY`.",
            "",
            mandatory_table(run),
            README_END,
        ]
    )


def update_readme(readme_path: Path, run: dict[str, Any]) -> bool:
    """Replace the metrics block in the README; returns False when markers are missing."""
    if not readme_path.exists():
        return False
    content = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(content):
        return False
    readme_path.write_text(pattern.sub(lambda _m: readme_block(run), content), encoding="utf-8")
    return True
