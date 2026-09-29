import pytest
from PIL import Image

from src.classifier.doc_type import DocTypeClassifier, classify_heuristic, heuristic_scores
from src.llm.client import ClaudeClient, LLMDisabledError, LLMOutputError
from tests.conftest import FakeAnthropic


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Invoice INV-001\nNIT: 900.111.222-3\nSubtotal: $100", "invoice"),
        (
            "MASTER SERVICES AGREEMENT\nParties: Acme & Northwind\nEffective Date: 2026-01-01",
            "contract",
        ),
        ("Receipt R-001\nPayment received: $42.50\nCashier: 3", "receipt"),
        ("APPLICATION FORM\nForm ID: F-1\nSubmitted by: Ana\nSignature:", "form"),
        ("ANNUAL REPORT 2025\nExecutive Summary\nFiscal Year: 2025", "report"),
    ],
)
def test_heuristic_types(text: str, expected: str) -> None:
    result = classify_heuristic(text)
    assert result.doc_type == expected
    assert result.method == "heuristic"
    assert 0.35 <= result.confidence <= 0.95


def test_heuristic_defaults_to_form_with_low_confidence() -> None:
    result = classify_heuristic("lorem ipsum dolor sit amet")
    assert result.doc_type == "form" and result.confidence == 0.35
    assert all(v == 0 for v in heuristic_scores("lorem ipsum").values())


async def test_classifier_without_llm_uses_heuristic() -> None:
    clf = DocTypeClassifier(None)
    assert not clf.llm_enabled
    assert (await clf.classify("Invoice 1 subtotal")).method == "heuristic"


async def test_classifier_with_claude(llm: ClaudeClient, fake_anthropic: FakeAnthropic) -> None:
    fake_anthropic.queue('{"type": "contract", "confidence": 0.93}')
    result = await DocTypeClassifier(llm).classify("some agreement text")
    assert result.doc_type == "contract" and result.method == "claude"
    assert result.confidence == pytest.approx(0.93)
    assert result.usage.llm_calls == 1


async def test_classifier_falls_back_on_bad_output(
    llm: ClaudeClient, fake_anthropic: FakeAnthropic
) -> None:
    fake_anthropic.queue('{"type": "spreadsheet"}')
    result = await DocTypeClassifier(llm).classify("Invoice INV-1 subtotal tax")
    assert result.method == "heuristic" and result.doc_type == "invoice"
    assert result.confidence <= 0.6


async def test_classify_image(llm: ClaudeClient, fake_anthropic: FakeAnthropic) -> None:
    fake_anthropic.queue('{"type": "receipt", "confidence": "0.8"}')
    result = await DocTypeClassifier(llm).classify_image(Image.new("RGB", (8, 8)))
    assert result.doc_type == "receipt" and result.method == "claude_vision"
    content = fake_anthropic.messages.create.call_args.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    fake_anthropic.queue('{"type": "receipt", "confidence": "high"}')
    with pytest.raises(LLMOutputError):
        await DocTypeClassifier(llm).classify_image(Image.new("RGB", (8, 8)))


async def test_classify_image_requires_client() -> None:
    with pytest.raises(LLMDisabledError):
        await DocTypeClassifier(None).classify_image(Image.new("RGB", (8, 8)))
