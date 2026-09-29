# Resultados de evaluación

Generado por `python -m eval.run` el 2026-09-29T01:39:02+00:00 · corrida `eval/runs/2026-09-29-heuristic-baseline.json` · commit `ccffa0d`.

> **Lectura obligatoria.** Solo la fila *Unstructured only* fue medida en esta corrida: es el **fallback determinista, sin LLM** (pdfplumber/OCR + reglas). No representa la calidad del sistema completo. Las filas con Claude Vision requieren `ANTHROPIC_API_KEY` y quedan pendientes; el código que las produce está implementado y probado con mocks.

Dataset: 105 documentos (100 sintéticos generados con semilla 20260929 + 5 formularios FUNSD reales de `data/funsd_sample/`), 30 tablas sintéticas en PDF. Los datos sintéticos están marcados como tales en `data/eval/test_set.jsonl` (`source`).

## Tabla obligatoria

| Pipeline | Field F1 | Doc Accuracy | Table F1 | Latencia | Costo/doc |
|---|---|---|---|---|---|
| Unstructured only (heurístico: pdfplumber/OCR + reglas, sin LLM) | 82.9 % | 71.0 % | 96.7 % | 18 ms (p95 41 ms) | $0.0000 |
| Unstructured + Claude Vision fallback (este) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) |
| Claude Vision directo | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) |

Field F1 es micro-F1 sobre pares (campo, valor) contra la verdad de terreno; Doc Accuracy exige todos los campos críticos correctos; Table F1 es F1 a nivel de celda (posicional) sobre las tablas sintéticas; latencia y costo se miden por documento. La fila heurística usa el modo `pdf` (PDF real vía pdfplumber); el modo `text` (texto ya extraído) es el techo de las reglas y se muestra aparte.

## Fila heurística por modo de entrada

| Modo | Docs | Field F1 (micro) | Field F1 (macro) | Doc Accuracy | Clasificador | Latencia media | p95 |
|---|---|---|---|---|---|---|---|
| `text` | 105 | 83.4 % | 81.4 % | 71.0 % | 99.1 % (105) | 5 ms | 7 ms |
| `pdf` | 105 | 82.9 % | 81.2 % | 71.0 % | 99.1 % (105) | 18 ms | 41 ms |
| `scanned` | 105 | 81.0 % | 79.1 % | 71.0 % | 99.1 % (105) | 688 ms | 1480 ms |

### Field F1 por tipo de documento (modo text)

| Campo | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| `contract` | 100.0 % | 67.9 % | 80.8 % | 95 | 0 | 45 |
| `form` | 100.0 % | 80.0 % | 88.9 % | 64 | 0 | 16 |
| `invoice` | 94.8 % | 64.5 % | 76.8 % | 129 | 7 | 71 |
| `receipt` | 100.0 % | 73.3 % | 84.6 % | 88 | 0 | 32 |
| `report` | 100.0 % | 90.0 % | 94.7 % | 72 | 0 | 8 |

### Field F1 por tipo de campo (modo text)

| Campo | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| `author` | 100.0 % | 100.0 % | 100.0 % | 20 | 0 | 0 |
| `contract_number` | 100.0 % | 55.0 % | 71.0 % | 11 | 0 | 9 |
| `currency` | 100.0 % | 100.0 % | 100.0 % | 60 | 0 | 0 |
| `date` | 100.0 % | 90.0 % | 94.7 % | 36 | 0 | 4 |
| `department` | 100.0 % | 80.0 % | 88.9 % | 16 | 0 | 4 |
| `due_date` | 100.0 % | 60.0 % | 75.0 % | 12 | 0 | 8 |
| `effective_date` | 100.0 % | 55.0 % | 71.0 % | 11 | 0 | 9 |
| `fiscal_year` | 100.0 % | 60.0 % | 75.0 % | 12 | 0 | 8 |
| `form_id` | 100.0 % | 80.0 % | 88.9 % | 16 | 0 | 4 |
| `governing_law` | 100.0 % | 55.0 % | 71.0 % | 11 | 0 | 9 |
| `invoice_number` | 100.0 % | 60.0 % | 75.0 % | 12 | 0 | 8 |
| `issue_date` | 100.0 % | 60.0 % | 75.0 % | 24 | 0 | 16 |
| `parties` | 100.0 % | 100.0 % | 100.0 % | 20 | 0 | 0 |
| `payment_method` | 100.0 % | 100.0 % | 100.0 % | 20 | 0 | 0 |
| `payment_terms` | 100.0 % | 60.0 % | 75.0 % | 12 | 0 | 8 |
| `receipt_number` | 100.0 % | 60.0 % | 75.0 % | 12 | 0 | 8 |
| `submitter` | 100.0 % | 80.0 % | 88.9 % | 16 | 0 | 4 |
| `subtotal` | 100.0 % | 60.0 % | 75.0 % | 12 | 0 | 8 |
| `tax` | 100.0 % | 60.0 % | 75.0 % | 12 | 0 | 8 |
| `term_months` | 100.0 % | 55.0 % | 71.0 % | 11 | 0 | 9 |
| `title` | 100.0 % | 100.0 % | 100.0 % | 20 | 0 | 0 |
| `total` | 78.1 % | 62.5 % | 69.4 % | 25 | 7 | 15 |
| `total_value` | 100.0 % | 55.0 % | 71.0 % | 11 | 0 | 9 |
| `vendor` | 100.0 % | 60.0 % | 75.0 % | 24 | 0 | 16 |
| `vendor_tax_id` | 100.0 % | 60.0 % | 75.0 % | 12 | 0 | 8 |

### FUNSD: pares pregunta/respuesta (modo text)

Pares (question → answer) enlazados en las 5 anotaciones: F1 estricto 14.2 % (P 77.8 %, R 7.8 %) sobre 5 documentos. El baseline solo reconoce pares `clave: valor` en la misma línea del texto reconstruido; la mayoría de las respuestas FUNSD están en otra línea o en casillas, de ahí el recall bajo.

### Distribución de routing (modo text)

| Routing | Docs |
|---|---|
| `auto_approve` | 7 |
| `human_review` | 65 |
| `vlm_fallback` | 33 |

## Extracción de tablas (pdfplumber)

30 tablas sintéticas (20 con líneas, 10 sin líneas). Cell F1 global 96.7 % (P 96.6 %, R 96.9 %); con líneas 100.0 %, sin líneas 91.4 %. Tablas detectadas: 30/30. Latencia media 8 ms.

## Diez peores casos (fila heurística, modo pdf)

| # | Caso | Tipo | F1 | Campos fallidos | Hallazgos |
|---|---|---|---|---|---|
| 1 | `syn-frm-010` | form | 0.0 % | date, department, form_id, submitter | missing_required, missing_required, missing_required |
| 2 | `syn-frm-011` | form | 0.0 % | date, department, form_id, submitter | missing_required, missing_required, missing_required |
| 3 | `syn-frm-014` | form | 0.0 % | date, department, form_id, submitter | missing_required, missing_required, missing_required |
| 4 | `syn-frm-018` | form | 0.0 % | date, department, form_id, submitter | missing_required, missing_required, missing_required |
| 5 | `syn-inv-001` | invoice | 15.4 % | due_date, invoice_number, issue_date, payment_terms, subtotal, tax, total, vendor, vendor_tax_id | missing_required, missing_required, missing_required, missing_required, missing_required |
| 6 | `syn-inv-002` | invoice | 15.4 % | due_date, invoice_number, issue_date, payment_terms, subtotal, tax, total, vendor, vendor_tax_id | missing_required, missing_required, missing_required, missing_required, missing_required |
| 7 | `syn-inv-007` | invoice | 15.4 % | due_date, invoice_number, issue_date, payment_terms, subtotal, tax, total, vendor, vendor_tax_id | missing_required, missing_required, missing_required, missing_required, missing_required |
| 8 | `syn-inv-011` | invoice | 15.4 % | due_date, invoice_number, issue_date, payment_terms, subtotal, tax, total, vendor, vendor_tax_id | missing_required, missing_required, missing_required, missing_required, missing_required |
| 9 | `syn-inv-016` | invoice | 15.4 % | due_date, invoice_number, issue_date, payment_terms, subtotal, tax, total, vendor, vendor_tax_id | missing_required, missing_required, missing_required, missing_required, missing_required |
| 10 | `syn-inv-009` | invoice | 16.7 % | due_date, invoice_number, issue_date, payment_terms, subtotal, tax, total, vendor, vendor_tax_id | missing_required, missing_required, missing_required, missing_required, missing_required, missing_required |

El análisis de causas está en `docs/error_analysis.md`.

## Pendiente (requiere llave o recursos externos)

- Filas *Unstructured + Claude Vision fallback* y *Claude Vision directo*: `ANTHROPIC_API_KEY`.
- LLM-as-judge (rúbrica numerada en `src/validators/llm_judge.py`): `ANTHROPIC_API_KEY`.
- FUNSD completo (199 formularios) y FinTabNet: descarga con `scripts/download_data.py` en una red sin restricciones.
- Document-level accuracy sobre DocVQA: cuenta en el portal RRC.
