"""Document type classifier — Claude with structured output + keyword heuristic."""
from __future__ import annotations

import json


HEURISTIC_KEYWORDS: dict[str, list[str]] = {
    "invoice":  ["invoice", "factura", "bill to", "vat", "nit"],
    "receipt":  ["receipt", "recibo", "payment received"],
    "contract": ["agreement", "contract", "parties", "effective date", "msa"],
    "form":     ["form id", "application form", "submission"],
    "report":   ["executive summary", "report", "fiscal year", "10-k"],
}


class DocTypeClassifier:
    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self.api_key = api_key

    async def classify(self, text: str) -> str:
        if not self.api_key:
            return self._heuristic(text)
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
        chat = ChatAnthropic(model=self.model, api_key=self.api_key, temperature=0, max_tokens=80, timeout=10.0)
        resp = await chat.ainvoke([
            SystemMessage(content='Classify this document. Return JSON only: {"type":"invoice|contract|form|report|receipt"}'),
            HumanMessage(content=text[:1500]),
        ])
        body = resp.content if isinstance(resp.content, str) else str(resp.content)
        body = body.strip()
        if body.startswith("```"):
            body = body.strip("`")
            if body.lower().startswith("json"):
                body = body[4:].lstrip()
        try:
            return json.loads(body).get("type", "form")
        except Exception:
            return self._heuristic(text)

    @staticmethod
    def _heuristic(text: str) -> str:
        t = text.lower()
        scores = {k: sum(1 for kw in kws if kw in t) for k, kws in HEURISTIC_KEYWORDS.items()}
        return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else "form"
