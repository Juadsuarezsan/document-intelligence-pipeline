import pytest
from PIL import Image

from src.extractors.heuristic import HeuristicExtractor
from src.extractors.llm_extractor import LLMExtractor, VLMExtractor, fields_from_model_output
from src.extractors.normalize import normalize_amount, normalize_date, normalize_text
from src.llm.client import ClaudeClient, LLMOutputError
from tests.conftest import FakeAnthropic


def _values(fields: list) -> dict[str, object]:  # type: ignore[type-arg]
    return {f.name: f.value for f in fields}


def test_normalize_dates() -> None:
    assert normalize_date("2026-03-14") == "2026-03-14"
    assert normalize_date("14/03/2026") == "2026-03-14"
    assert normalize_date("03/14/2026") == "2026-03-14"  # month > 12 swap
    assert normalize_date("March 14, 2026") == "2026-03-14"
    assert normalize_date("14 March 2026") == "2026-03-14"
    assert normalize_date("14 de marzo de 2026") == "2026-03-14"
    assert normalize_date("2026-13-40") is None
    assert normalize_date("no date") is None


def test_normalize_amounts_and_text() -> None:
    assert normalize_amount("2,499.00") == 2499.0
    assert normalize_amount("2.499,00") == 2499.0
    assert normalize_amount("USD 1,250") == 1250.0
    assert normalize_amount("$ -12.5") == -12.5
    assert normalize_amount(7) == 7.0
    assert normalize_amount("n/a") is None
    assert normalize_text("  ACME  Corp. ") == "acme corp"


def test_heuristic_invoice(invoice_text: str) -> None:
    got = _values(HeuristicExtractor().extract(invoice_text, "invoice").fields)
    assert got["invoice_number"] == "INV-2026-0042"
    assert got["vendor_tax_id"] == "900.123.456-9"
    assert got["issue_date"] == "2026-03-14"
    assert got["due_date"] == "2026-04-13"
    assert got["subtotal"] == 2100.0 and got["tax"] == 399.0 and got["total"] == 2499.0
    assert got["currency"] == "USD"
    assert got["payment_terms"] == "Net 30"
    assert got["vendor"] == "ACME CONSULTING S.A.S."


def test_heuristic_contract_prose(contract_text: str) -> None:
    got = _values(HeuristicExtractor().extract(contract_text, "contract").fields)
    assert got["contract_number"] == "MSA-2026-001"
    assert got["effective_date"] == "2026-03-01"
    assert got["term_months"] == 24
    assert got["total_value"] == 120000.0
    assert got["currency"] == "USD"
    assert got["governing_law"] == "State of Delaware"
    assert got["parties"] == "Acme Consulting S.A.S.; Globex Corporation"


def test_heuristic_receipt_form_report() -> None:
    receipt = "QUARRY LANE BAKERY\nReceipt #: R-123456\nDate: 04/03/2026\nTOTAL: EUR 45.20\nPaid by: MASTERCARD\n"
    got = _values(HeuristicExtractor().extract(receipt, "receipt").fields)
    assert got["receipt_number"] == "R-123456"
    assert got["total"] == 45.2 and got["currency"] == "EUR"
    assert got["payment_method"] == "mastercard"
    assert got["issue_date"] == "2026-03-04"

    form = "APPLICATION FORM\nForm ID: HR-1001\nName: Ana Pérez\nDepartment: Finance\nDate: 2026-02-02\n"
    got = _values(HeuristicExtractor().extract(form, "form").fields)
    assert got == {
        "form_id": "HR-1001",
        "submitter": "Ana Pérez",
        "date": "2026-02-02",
        "department": "Finance",
    }

    report = "MEMORANDUM\nTO: Board\nFROM: J. Doe\nDATE: May 1, 2025\nRE: Site Audit Report\nPeriod: FY2024\n"
    got = _values(HeuristicExtractor().extract(report, "report").fields)
    assert got["title"] == "Site Audit Report" and got["author"] == "J. Doe"
    assert got["date"] == "2025-05-01" and got["fiscal_year"] == "2024"


def test_heuristic_returns_empty_on_unrelated_text() -> None:
    assert (
        HeuristicExtractor().extract("Just some random prose with no fields", "invoice").fields
        == []
    )


def test_heuristic_tax_does_not_capture_tax_id() -> None:
    text = "INVOICE\nTax ID: 900123456\nInvoice #: A-1\nAmount Due: 10.00\n"
    got = _values(HeuristicExtractor().extract(text, "invoice").fields)
    assert got["vendor_tax_id"] == "900123456"
    assert "tax" not in got


def test_fields_from_model_output_validation() -> None:
    data = {
        "fields": [
            {"name": "invoice_number", "value": "X-1", "confidence": 0.95},
            {"name": "total", "value": "1,200.50", "confidence": 0.9},
            {"name": "issue_date", "value": "March 3, 2026", "confidence": 0.8},
            {"name": "unknown_field", "value": "drop me", "confidence": 0.5},
            {"name": "vendor", "value": None, "confidence": 0.1},
        ]
    }
    fields = fields_from_model_output(data, "invoice", "extractor")
    assert _values(fields) == {"invoice_number": "X-1", "total": 1200.5, "issue_date": "2026-03-03"}
    assert all(f.source == "extractor" for f in fields)
    with pytest.raises(LLMOutputError):
        fields_from_model_output({"fields": "nope"}, "invoice", "extractor")
    with pytest.raises(LLMOutputError):
        fields_from_model_output(
            {"fields": [{"name": "total", "value": 1, "confidence": 5}]}, "invoice", "extractor"
        )
    with pytest.raises(LLMOutputError):
        fields_from_model_output(
            {"fields": [{"name": "term_months", "value": "many"}]}, "contract", "extractor"
        )


async def test_llm_extractor_without_key_is_heuristic(invoice_text: str) -> None:
    result = await LLMExtractor(None).extract(invoice_text, "invoice")
    assert result.method == "heuristic" and result.usage.llm_calls == 0
    assert "invoice_number" in _values(result.fields)


async def test_llm_extractor_with_claude(
    llm: ClaudeClient, fake_anthropic: FakeAnthropic, invoice_text: str
) -> None:
    fake_anthropic.queue('{"fields": [{"name": "total", "value": 2499, "confidence": 0.97}]}')
    result = await LLMExtractor(llm).extract(invoice_text, "invoice")
    assert result.method == "claude" and result.usage.llm_calls == 1
    assert _values(result.fields) == {"total": 2499.0}


async def test_llm_extractor_falls_back_on_bad_output(
    llm: ClaudeClient, fake_anthropic: FakeAnthropic, invoice_text: str
) -> None:
    fake_anthropic.queue("I cannot do that")
    result = await LLMExtractor(llm).extract(invoice_text, "invoice")
    assert result.method == "heuristic_after_llm_error"
    assert "invoice_number" in _values(result.fields)


async def test_vlm_extractor(llm: ClaudeClient, fake_anthropic: FakeAnthropic) -> None:
    fake_anthropic.queue(
        '{"fields": [{"name": "receipt_number", "value": "R-9", "confidence": 0.9}]}'
    )
    result = await VLMExtractor(llm).extract([Image.new("RGB", (8, 8))], "receipt")
    assert result.method == "claude_vision"
    assert result.fields[0].source == "vlm" and result.fields[0].value == "R-9"
