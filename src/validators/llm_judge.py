"""LLM-as-judge coherence check with an explicit, numbered rubric.

The judge sees the document text and the extracted fields and scores the
extraction on five criteria. It only runs with an API key; the rubric lives
here in code so the evaluation is reproducible and reviewable.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from src.llm.client import ClaudeClient, LLMOutputError
from src.schemas.document import DocumentType, ExtractedField, Finding, UsageStats

RUBRIC: list[tuple[int, str, str]] = [
    (
        1,
        "grounding",
        "Every extracted value appears in the document text (verbatim or trivially "
        "normalised, e.g. '1,200.00' -> 1200.0). Hallucinated values score 0.",
    ),
    (
        2,
        "field_semantics",
        "Each value is placed under the correct field: the vendor's tax id is not "
        "under invoice_number, the due date is not under issue_date, etc.",
    ),
    (
        3,
        "completeness",
        "Fields that are clearly present in the text were extracted. Missing a "
        "visible required field costs points; absent-in-source fields are fine.",
    ),
    (
        4,
        "internal_consistency",
        "Numeric and temporal relations hold: subtotal + tax = total, due_date >= "
        "issue_date, term matches the stated period.",
    ),
    (
        5,
        "confidence_calibration",
        "Reported confidences are sensible: verbatim values near 0.95+, inferred "
        "ones lower, no 0.99 on a guess.",
    ),
]

RUBRIC_EXAMPLES = (
    "Example PASS: text says 'Total Due: USD 1,250.00' and the field is total=1250.0, "
    "currency='USD' with confidence 0.95.\n"
    "Example FAIL (grounding): text has no tax line but tax=190.0 was extracted.\n"
    "Example FAIL (field_semantics): 'NIT 900.123.456-1' placed in invoice_number."
)


def rubric_text() -> str:
    """Render the numbered rubric for the prompt (and for the docs)."""
    lines = [f"{n}. {name}: {desc}" for n, name, desc in RUBRIC]
    return "\n".join(lines)


JUDGE_SYSTEM = f"""You are a strict reviewer of automated document extraction.

Score the extraction against this rubric, each criterion from 0 to 2
(0 = fails, 1 = partially, 2 = fully satisfied):
{rubric_text()}

{RUBRIC_EXAMPLES}

Return ONLY JSON:
{{"scores": {{"grounding": 0-2, "field_semantics": 0-2, "completeness": 0-2,
             "internal_consistency": 0-2, "confidence_calibration": 0-2}},
  "issues": ["short description of each concrete problem"],
  "verdict": "pass" | "review" | "fail"}}
verdict is "pass" when total >= 8, "review" for 5-7, "fail" below 5.
"""


class JudgeVerdict(BaseModel):
    """Structured judge output."""

    scores: dict[str, int]
    issues: list[str] = Field(default_factory=list)
    verdict: str
    total: int
    usage: UsageStats = UsageStats()

    @property
    def normalized(self) -> float:
        """Total score scaled to 0-1."""
        return self.total / (2 * len(RUBRIC))


def parse_verdict(data: dict[str, Any], usage: UsageStats) -> JudgeVerdict:
    """Validate the judge JSON.

    Raises:
        LLMOutputError: If scores are missing or out of range.
    """
    raw_scores = data.get("scores")
    if not isinstance(raw_scores, dict):
        raise LLMOutputError("judge output missing 'scores'")
    scores: dict[str, int] = {}
    for _, name, _ in RUBRIC:
        try:
            value = int(raw_scores.get(name, -1))
        except (TypeError, ValueError) as exc:
            raise LLMOutputError(f"judge score {name} not an int") from exc
        if value not in (0, 1, 2):
            raise LLMOutputError(f"judge score {name} out of range: {value}")
        scores[name] = value
    total = sum(scores.values())
    verdict = "pass" if total >= 8 else "review" if total >= 5 else "fail"
    issues = (
        [str(i) for i in data.get("issues", [])] if isinstance(data.get("issues"), list) else []
    )
    return JudgeVerdict(scores=scores, issues=issues, verdict=verdict, total=total, usage=usage)


class CoherenceJudge:
    """Runs the rubric with Claude."""

    def __init__(self, llm: ClaudeClient, max_chars: int = 8000) -> None:
        self._llm = llm
        self._max_chars = max_chars

    @property
    def enabled(self) -> bool:
        """True when the client has credentials."""
        return self._llm.enabled

    async def judge(
        self, text: str, fields: list[ExtractedField], doc_type: DocumentType
    ) -> JudgeVerdict:
        """Score an extraction.

        Raises:
            LLMDisabledError: Without an API key.
            LLMOutputError: On unparsable model output.
        """
        payload = "\n".join(f"- {f.name} = {f.value!r} (confidence {f.confidence})" for f in fields)
        content = (
            f"Document type: {doc_type}\n\nDOCUMENT TEXT:\n{text[: self._max_chars]}\n\n"
            f"EXTRACTED FIELDS:\n{payload or '(none)'}"
        )
        data, result = await self._llm.complete_json(
            system=JUDGE_SYSTEM, content=content, max_tokens=600
        )
        verdict = parse_verdict(data, result.usage)
        logger.info("judge verdict={} total={}", verdict.verdict, verdict.total)
        return verdict

    @staticmethod
    def to_findings(verdict: JudgeVerdict) -> list[Finding]:
        """Translate a verdict into validation findings."""
        if verdict.verdict == "pass":
            return []
        severity = "error" if verdict.verdict == "fail" else "warn"
        return [
            Finding(
                severity=severity,
                code="judge_" + verdict.verdict,
                message="; ".join(verdict.issues) or "judge flagged the extraction",
                details={"total": verdict.total, **verdict.scores},
            )
        ]
