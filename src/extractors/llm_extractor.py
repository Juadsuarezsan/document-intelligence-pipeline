"""Claude-based field extractor with strict Pydantic-validated JSON output."""
from __future__ import annotations

import json
from tenacity import retry, stop_after_attempt, wait_exponential

from src.api.schemas import ExtractedField
from src.extractors.schemas_by_type import REQUIRED_FIELDS


SYSTEM_TEMPLATE = """You extract structured fields from {doc_type} documents.

For each REQUIRED field, output a JSON object:
  {{
    "name": "<field_name>",
    "value": <value or null if missing>,
    "confidence": 0.0 - 1.0
  }}

Required fields for {doc_type}:
{required_list}

Output ONLY JSON with this exact shape:
{{ "fields": [ {{...}}, ... ] }}

Confidence guidelines:
- 0.95+ : value appears verbatim in source
- 0.80-0.95 : value derived but high-confidence (e.g. subtotal+tax → total)
- 0.60-0.80 : best inference but ambiguity exists
- < 0.60 : guess; consider returning null instead

Values:
- numbers as numbers (no currency symbols)
- dates in ISO-8601 (YYYY-MM-DD)
- IDs as strings exactly as written
"""


class LLMExtractor:
    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self.api_key = api_key

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
    async def extract(self, text: str, doc_type: str) -> list[ExtractedField]:
        if not self.api_key:
            return self._heuristic_extract(text, doc_type)
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
        required = REQUIRED_FIELDS.get(doc_type, REQUIRED_FIELDS["form"])
        sys = SYSTEM_TEMPLATE.format(
            doc_type=doc_type, required_list="\n".join(f"  - {f}" for f in required),
        )
        chat = ChatAnthropic(model=self.model, api_key=self.api_key, temperature=0, max_tokens=1500, timeout=20.0)
        resp = await chat.ainvoke([SystemMessage(content=sys), HumanMessage(content=text[:8000])])
        body = resp.content if isinstance(resp.content, str) else str(resp.content)
        body = body.strip()
        if body.startswith("```"):
            body = body.strip("`")
            if body.lower().startswith("json"):
                body = body[4:].lstrip()
        try:
            raw = json.loads(body)
            return [ExtractedField(**f, source="extractor") for f in raw.get("fields", [])]
        except Exception:
            return self._heuristic_extract(text, doc_type)

    @staticmethod
    def _heuristic_extract(text: str, doc_type: str) -> list[ExtractedField]:
        """Regex-based fallback. Limited but works without API key."""
        import re
        fields: list[ExtractedField] = []
        # Invoice / receipt heuristics
        if doc_type in ("invoice", "receipt"):
            inv_match = re.search(r"(?:invoice|receipt)[\s#:]+([A-Z0-9\-]+)", text, re.IGNORECASE)
            if inv_match:
                key = "invoice_number" if doc_type == "invoice" else "receipt_number"
                fields.append(ExtractedField(name=key, value=inv_match.group(1), confidence=0.85, source="regex"))
            tax_match = re.search(r"(?:NIT|Tax ID|VAT)[:\s]+([\d\.\-A-Z]+)", text, re.IGNORECASE)
            if tax_match:
                fields.append(ExtractedField(name="vendor_tax_id", value=tax_match.group(1).strip(),
                                              confidence=0.80, source="regex"))
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
            if date_match:
                fields.append(ExtractedField(name="issue_date", value=date_match.group(1),
                                              confidence=0.90, source="regex"))
            total_match = re.search(r"Total[^\d]*([\d,]+\.\d{2})", text, re.IGNORECASE)
            if total_match:
                try:
                    val = float(total_match.group(1).replace(",", ""))
                    fields.append(ExtractedField(name="total", value=val, confidence=0.85, source="regex"))
                except ValueError:
                    pass
            cur_match = re.search(r"\b(USD|EUR|COP|GBP|MXN)\b", text)
            if cur_match:
                fields.append(ExtractedField(name="currency", value=cur_match.group(1), confidence=0.95, source="regex"))
        return fields
