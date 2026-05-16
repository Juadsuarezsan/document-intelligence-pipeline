from src.api.schemas import ExtractedField
from src.validators.cross_field import (
    check_invoice_math, check_iso_dates, check_required, routing_decision, validate_all,
)


def _f(name: str, value, conf: float = 0.9) -> ExtractedField:
    return ExtractedField(name=name, value=value, confidence=conf, source="extractor")


def test_missing_required_detected():
    fields = [_f("invoice_number", "X-1"), _f("vendor", "Acme")]
    findings = check_required(fields, "invoice")
    codes = {f["code"] for f in findings}
    assert codes == {"missing_required"}
    missing_names = {f["field"] for f in findings}
    assert "currency" in missing_names
    assert "total" in missing_names


def test_iso_date_warning():
    findings = check_iso_dates([_f("issue_date", "March 14, 2026")])
    assert len(findings) == 1
    assert findings[0]["code"] == "non_iso_date"


def test_invoice_math_pass():
    fields = [_f("subtotal", 100.0), _f("tax", 19.0), _f("total", 119.0)]
    assert check_invoice_math(fields) == []


def test_invoice_math_fail():
    fields = [_f("subtotal", 100.0), _f("tax", 19.0), _f("total", 200.0)]
    findings = check_invoice_math(fields)
    assert len(findings) == 1
    assert findings[0]["code"] == "math_mismatch"


def test_validate_all_penalizes_errors():
    fields = [_f("subtotal", 100.0), _f("tax", 19.0), _f("total", 200.0, conf=0.95)]
    conf, findings = validate_all(fields, "invoice")
    # Math mismatch should knock confidence down
    assert conf < 0.95


def test_routing_decision():
    assert routing_decision(0.95, auto_thr=0.90, review_thr=0.70) == "auto_approve"
    assert routing_decision(0.80, auto_thr=0.90, review_thr=0.70) == "human_review"
    assert routing_decision(0.50, auto_thr=0.90, review_thr=0.70) == "vlm_fallback"
