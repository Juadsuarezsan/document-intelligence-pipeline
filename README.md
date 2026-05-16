# Project 04 — Document Intelligence Pipeline

> Intelligent Document Processing for complex PDFs: contracts, financial reports, medical forms, scanned forms with tables. Layout analysis → OCR → Claude Vision fallback → Pydantic-validated extraction → confidence-gated routing.

[![Status](https://img.shields.io/badge/status-planned-fbbf24)]()
[![LLM](https://img.shields.io/badge/LLM-Claude%20Vision-7c5cff)]()
[![Domain](https://img.shields.io/badge/domain-Document%20AI-22d3ee)]()

**Industrial use case:** Hyperscience, Rossum, Klarity — IDP for legal-tech, fintech, healthcare, insurance.

## What this project does

Extracts structured fields from messy real-world PDFs. Routes simple text PDFs through `unstructured`/`docling` (fast, cheap), scanned ones through OCR (Tesseract/PaddleOCR), and the hard ones through Claude Vision. Validates extractions with Pydantic schemas + cross-field rules. Auto-approves high-confidence; routes the rest to a human reviewer.

## Architecture

```
PDF input
   │
   ▼
[Document Classifier] Claude Vision → type: contract | invoice | form | report
   │
   ▼
[Layout Analyzer]
   ├─ text-based → unstructured / docling
   ├─ scanned   → OCR (Tesseract or PaddleOCR)
   └─ hard      → Claude Vision direct
   │
   ▼
[Table Extractor] Camelot or Table Transformer (TATR)
   │
   ▼
[Field Extractor] Claude + Pydantic schema per document type
   │
   ▼
[Validator]
   ├─ regex (NIT, dates, amounts)
   ├─ cross-field (subtotal + tax = total)
   └─ LLM-as-judge for coherence
   │
   ▼
[Confidence Scorer] per-field + global
   ├─ > 0.9 → auto-approve
   ├─ 0.7-0.9 → human review queue
   └─ < 0.7 → re-process with more expensive VLM
```

## Roadmap to v1.0.0

1. [ ] Download FUNSD (199 scanned forms) + DocVQA (12K docs) + PubLayNet (360K pages)
2. [ ] Document type classifier (Claude Vision)
3. [ ] Three-path layout analyzer (unstructured | OCR | Vision)
4. [ ] Table extractor (Camelot for text PDFs, TATR for image-based)
5. [ ] Pydantic schemas per document type (invoice, contract, form, report)
6. [ ] Regex + cross-field validators
7. [ ] Confidence scorer + routing
8. [ ] Eval: field-level F1 on FUNSD, doc-level accuracy on DocVQA, table F1 on FinTabNet
9. [ ] Next.js demo with PDF viewer side-by-side with JSON output + confidence per field
10. [ ] Gallery of 10 pre-processed documents accessible from the demo

## Stack

| Layer | Technology |
|---|---|
| Parsing | unstructured 0.16+ or docling (IBM, 2024) |
| OCR | Tesseract 5 or PaddleOCR |
| VLM | Claude Sonnet 4.5 with vision input |
| Tables | Camelot or Table Transformer |
| Layout | LayoutParser or detectron2 |
| Schemas | Pydantic v2 |
| Reasoning | Claude Sonnet 4.5 |
| Storage | PostgreSQL |
| Frontend | Next.js with `react-pdf` viewer |
| Observability | LangSmith |

## Definition of Done — project-specific

- [ ] Pipeline works on all 3 input types: clean text, scanned, complex layout
- [ ] Field-level F1 reported on FUNSD test set
- [ ] Document-level accuracy reported on DocVQA
- [ ] Table extraction evaluated on FinTabNet or equivalent
- [ ] Demo: upload a PDF, see extraction with confidence scores in real time
- [ ] Gallery of 10 pre-processed sample documents linked from demo
- [ ] Comparative table: unstructured-only vs unstructured + Vision fallback vs Vision direct

Plus the 12 universal DoD blocks.

## License

MIT.
