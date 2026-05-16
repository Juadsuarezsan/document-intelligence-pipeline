"""End-to-end IDP eval — field-level precision/recall + finding detection rate."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from src.api.schemas import ExtractRequest
from src.config import get_settings
from src.extractors.llm_extractor import LLMExtractor
from src.parsers.text_parser import parse_text
from src.validators.cross_field import routing_decision, validate_all

DATA = Path(__file__).parent.parent.parent / "data" / "eval" / "cases.jsonl"


async def run_eval() -> dict[str, Any]:
    s = get_settings()
    extractor = LLMExtractor(model=s.anthropic_model, api_key=s.anthropic_api_key)
    cases = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]

    field_hits = 0
    field_total = 0
    finding_hits = 0
    finding_total = 0
    confs: list[float] = []
    routings: dict[str, int] = {"auto_approve": 0, "human_review": 0, "vlm_fallback": 0}
    per_case: list[dict[str, Any]] = []

    for case in cases:
        fields = await extractor.extract(case["text"], case["document_type"])
        confidence, findings = validate_all(fields, case["document_type"])
        route = routing_decision(confidence,
                                  auto_thr=s.confidence_auto_approve,
                                  review_thr=s.confidence_human_review)
        confs.append(confidence)
        routings[route] = routings.get(route, 0) + 1

        gt = case.get("ground_truth", {})
        expected_finding = gt.get("expected_finding")
        if expected_finding:
            finding_total += 1
            if any(f["code"] == expected_finding for f in findings):
                finding_hits += 1
        else:
            # Score field-level precision against the ground truth
            extracted = {f.name: f.value for f in fields if f.value is not None}
            for k, v in gt.items():
                if k == "expected_finding":
                    continue
                field_total += 1
                got = extracted.get(k)
                if isinstance(v, float) and isinstance(got, (int, float)):
                    if abs(float(got) - v) < 0.01:
                        field_hits += 1
                elif str(got) == str(v):
                    field_hits += 1

        per_case.append({
            "id": case["id"],
            "doc_type": case["document_type"],
            "confidence": confidence,
            "routing": route,
            "fields_extracted": len(fields),
            "findings": findings,
        })

    n = len(cases)
    return {
        "n": n,
        "field_accuracy":         field_hits / field_total if field_total else 0.0,
        "finding_detection_rate": finding_hits / finding_total if finding_total else 1.0,
        "avg_confidence":         sum(confs) / n if n else 0.0,
        "routing_distribution":   routings,
        "cases":                  per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(run_eval())
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"Cases:           {report['n']}")
        print(f"Field accuracy:  {report['field_accuracy']:.1%}")
        print(f"Finding rate:    {report['finding_detection_rate']:.1%}")
        print(f"Avg confidence:  {report['avg_confidence']:.2f}")
        print(f"Routing:         {report['routing_distribution']}")
    if report["finding_detection_rate"] < 0.5:
        sys.exit(1)


if __name__ == "__main__":
    main()
