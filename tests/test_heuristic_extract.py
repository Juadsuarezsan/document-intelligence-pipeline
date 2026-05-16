import pytest
from src.extractors.llm_extractor import LLMExtractor


@pytest.fixture
def extractor():
    return LLMExtractor(model="x", api_key=None)


@pytest.mark.asyncio
async def test_extract_invoice_fields(extractor):
    text = (
        "ACME CORP\nNIT: 900.123.456-1\nInvoice: INV-2026-0042\n"
        "Date: 2026-03-14\nTotal: 2,499.00\nCurrency: USD"
    )
    fields = await extractor.extract(text, "invoice")
    names = {f.name for f in fields}
    assert "invoice_number" in names
    assert "vendor_tax_id" in names
    assert "issue_date" in names
    assert "currency" in names


@pytest.mark.asyncio
async def test_heuristic_returns_empty_on_unrelated_text(extractor):
    fields = await extractor.extract("Just some random prose with no fields", "invoice")
    assert fields == []
