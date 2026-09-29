import pytest

from src.llm.client import ClaudeClient, LLMOutputError
from src.schemas.document import ExtractedField, UsageStats
from src.scoring.confidence import (
    coverage,
    document_confidence,
    mean_field_confidence,
    penalize_fields,
    routing_decision,
)
from src.validators.cross_field import (
    check_contract_term,
    check_date_order,
    check_invoice_math,
    check_required,
    validate_fields,
)
from src.validators.llm_judge import RUBRIC, CoherenceJudge, parse_verdict, rubric_text
from src.validators.pii import redact_fields, redact_text
from src.validators.regex_rules import (
    check_amounts,
    check_iso_dates,
    check_tax_id,
    is_valid_nit,
    nit_check_digit,
)
from tests.conftest import FakeAnthropic


def _f(name: str, value: object, conf: float = 0.9) -> ExtractedField:
    return ExtractedField(name=name, value=value, confidence=conf, source="extractor")  # type: ignore[arg-type]


def test_nit_check_digit_known_values() -> None:
    # 900.123.456 -> check digit computed with the DIAN weights
    d = nit_check_digit("900123456")
    assert 0 <= d <= 9
    assert is_valid_nit(f"900.123.456-{d}")
    assert not is_valid_nit(f"900.123.456-{(d + 1) % 10}")
    assert not is_valid_nit("abc")
    assert (
        check_tax_id([_f("vendor_tax_id", f"900123456-{(d + 1) % 10}")])[0].code
        == "invalid_nit_check_digit"
    )
    assert check_tax_id([_f("vendor_tax_id", "??")])[0].code == "implausible_tax_id"
    assert check_tax_id([_f("vendor_tax_id", "DE123456789")]) == []


def test_missing_required_detected() -> None:
    findings = check_required([_f("invoice_number", "X-1"), _f("vendor", "Acme")], "invoice")
    assert {f.code for f in findings} == {"missing_required"}
    assert {"currency", "total"} <= {f.field for f in findings}


def test_iso_date_rules() -> None:
    assert check_iso_dates([_f("issue_date", "March 14, 2026")])[0].code == "non_iso_date"
    assert check_iso_dates([_f("date", "2026-02-30")])[0].code == "invalid_date"
    assert check_iso_dates([_f("issue_date", "2026-02-28")]) == []


def test_amount_rules() -> None:
    codes = {
        f.code for f in check_amounts([_f("total", "abc"), _f("tax", -1.0), _f("currency", "usd")])
    }
    assert codes == {"non_numeric_amount", "negative_amount", "invalid_currency_code"}


def test_invoice_math() -> None:
    assert check_invoice_math([_f("subtotal", 100.0), _f("tax", 19.0), _f("total", 119.0)]) == []
    bad = check_invoice_math([_f("subtotal", 100.0), _f("tax", 19.0), _f("total", 200.0)])
    assert bad[0].code == "math_mismatch" and bad[0].details["expected_total"] == 119.0
    assert check_invoice_math([_f("subtotal", 100.0)]) == []


def test_date_order_and_term() -> None:
    assert (
        check_date_order([_f("issue_date", "2026-03-01"), _f("due_date", "2026-02-01")])[0].code
        == "due_before_issue"
    )
    assert check_date_order([_f("issue_date", "2026-03-01"), _f("due_date", "bad")]) == []
    assert check_contract_term([_f("term_months", 0)])[0].code == "implausible_term"
    assert check_contract_term([_f("term_months", 24)]) == []


def test_validate_fields_runs_type_specific_checks() -> None:
    codes = {
        f.code
        for f in validate_fields([_f("subtotal", 1.0), _f("tax", 1.0), _f("total", 5.0)], "invoice")
    }
    assert "math_mismatch" in codes and "missing_required" in codes


def test_confidence_and_routing() -> None:
    fields = [_f("subtotal", 100.0), _f("tax", 19.0), _f("total", 200.0, conf=0.95)]
    findings = validate_fields(fields, "invoice")
    assert mean_field_confidence([]) == 0.0
    assert coverage(fields, "invoice") == pytest.approx(3 / 8)
    penalized = penalize_fields(fields, findings)
    assert penalized[2].confidence < 0.95
    conf = document_confidence(fields, findings, "invoice")
    assert 0.0 <= conf < 0.5
    assert routing_decision(0.95, auto_thr=0.90, review_thr=0.70) == "auto_approve"
    assert routing_decision(0.80, auto_thr=0.90, review_thr=0.70) == "human_review"
    assert routing_decision(0.50, auto_thr=0.90, review_thr=0.70) == "vlm_fallback"


def test_pii_redaction() -> None:
    text = "Contact ana@example.org or +57 300 123 4567; SSN 123-45-6789; card 4111 1111 1111 1111; ref 2026-0042"
    masked, report = redact_text(text)
    assert "[REDACTED:email]" in masked and "[REDACTED:ssn]" in masked
    assert "[REDACTED:credit_card]" in masked and "[REDACTED:phone]" in masked
    assert "2026-0042" in masked
    assert report.total == 4
    fields, rep = redact_fields(
        [_f("submitter", "Ana ana@example.org"), _f("invoice_number", "123-45-6789")]
    )
    assert fields[0].redacted and fields[0].value == "Ana [REDACTED:email]"
    assert not fields[1].redacted and rep.counts == {"email": 1}


def test_rubric_is_numbered_and_parse_verdict() -> None:
    assert [n for n, _, _ in RUBRIC] == [1, 2, 3, 4, 5]
    assert rubric_text().startswith("1. grounding")
    verdict = parse_verdict({"scores": {n: 2 for _, n, _ in RUBRIC}, "issues": []}, UsageStats())
    assert verdict.verdict == "pass" and verdict.total == 10 and verdict.normalized == 1.0
    low = parse_verdict(
        {"scores": {n: 0 for _, n, _ in RUBRIC}, "issues": ["hallucinated total"]}, UsageStats()
    )
    assert low.verdict == "fail"
    assert CoherenceJudge.to_findings(low)[0].severity == "error"
    assert CoherenceJudge.to_findings(verdict) == []
    with pytest.raises(LLMOutputError):
        parse_verdict({"scores": {"grounding": 7}}, UsageStats())
    with pytest.raises(LLMOutputError):
        parse_verdict({"nope": 1}, UsageStats())


async def test_judge_with_mocked_claude(llm: ClaudeClient, fake_anthropic: FakeAnthropic) -> None:
    fake_anthropic.queue(
        '{"scores": {"grounding": 2, "field_semantics": 2, "completeness": 1, "internal_consistency": 1, "confidence_calibration": 1}, "issues": ["tax missing"]}'
    )
    verdict = await CoherenceJudge(llm).judge("Invoice text", [_f("total", 10.0)], "invoice")
    assert verdict.verdict == "review" and verdict.total == 7
    assert verdict.usage.llm_calls == 1
    system = fake_anthropic.messages.create.call_args.kwargs["system"]
    assert "1. grounding" in system and "Example FAIL" in system
