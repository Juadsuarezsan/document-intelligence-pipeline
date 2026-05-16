import pytest
from src.classifier.doc_type import DocTypeClassifier


@pytest.fixture
def classifier():
    return DocTypeClassifier(model="x", api_key=None)


@pytest.mark.asyncio
async def test_invoice_keywords(classifier):
    out = await classifier.classify("Invoice INV-001\nNIT: 900.111.222-3\nSubtotal: $100")
    assert out == "invoice"


@pytest.mark.asyncio
async def test_contract_keywords(classifier):
    out = await classifier.classify("MASTER SERVICES AGREEMENT\nParties: Acme & Northwind\nEffective Date: 2026-01-01")
    assert out == "contract"


@pytest.mark.asyncio
async def test_receipt_keywords(classifier):
    out = await classifier.classify("Receipt R-001\nPayment received: $42.50")
    assert out == "receipt"
