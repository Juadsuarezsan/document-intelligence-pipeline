"""Pre-bake the demo gallery: ``python scripts/build_demo_gallery.py``.

Runs the heuristic pipeline (no API key, deterministic) over the five FUNSD
sample forms and a fixed selection of synthetic documents, then writes
``demo/predictions.json`` with the extracted JSON, per-field confidence,
validation findings, routing and latency. The demo page renders this file and
labels every entry as produced by the deterministic fallback.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings  # noqa: E402
from src.eval.dataset import ROOT, EvalCase, load_cases  # noqa: E402
from src.eval.pipelines import case_pdf  # noqa: E402
from src.pipeline.graph import DocumentPipeline, build_deps  # noqa: E402

OUT = ROOT / "demo" / "predictions.json"
GALLERY_IDS = [
    "funsd-0000971160",
    "funsd-0000989556",
    "funsd-0000990274",
    "funsd-0000999294",
    "funsd-0001118259",
    "syn-inv-001",
    "syn-inv-004",
    "syn-rcp-002",
    "syn-ctr-003",
    "syn-ctr-001",
    "syn-frm-002",
    "syn-rpt-001",
]


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


async def build(cases: list[EvalCase]) -> list[dict[str, Any]]:
    """Run every gallery case through the heuristic pipeline (PDF mode for synthetic)."""
    settings = Settings(ANTHROPIC_API_KEY=None, USE_CLAUDE_VISION_FALLBACK=False)
    pipeline = DocumentPipeline(build_deps(settings))
    items: list[dict[str, Any]] = []
    for case in cases:
        if case.source == "funsd":
            result = await pipeline.run(text=case.text)
            preview = case.text
        else:
            result = await pipeline.run(pdf_bytes=case_pdf(case))
            preview = case.text
        items.append(
            {
                "id": case.id,
                "source": case.source,
                "label": (
                    f"FUNSD form {case.id.split('-')[1]} (real annotation)"
                    if case.source == "funsd"
                    else f"Synthetic {case.doc_type} · template {case.template}"
                ),
                "expected_type": case.doc_type,
                "text_preview": preview[:1800],
                "ground_truth": case.ground_truth,
                "result": result.model_dump(mode="json"),
            }
        )
    return items


def main() -> int:
    """Write ``demo/predictions.json``."""
    by_id = {c.id: c for c in load_cases()}
    cases = [by_id[i] for i in GALLERY_IDS if i in by_id]
    items = asyncio.run(build(cases))
    payload = {
        "project": "document-intelligence-pipeline",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "pipeline": "heuristic fallback (deterministic, no LLM) - regenerate with make demo",
        "model_for_llm_paths": "claude-sonnet-4-5-20250929",
        "items": items,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(items)} gallery items to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
