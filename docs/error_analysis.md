# Análisis de errores

Corrida: `eval/runs/2026-09-29-heuristic-baseline.json`, fila heurística
(fallback determinista, sin LLM), modo `pdf`. Field F1 micro 82,9 %, Doc
Accuracy 71,0 %, Table F1 96,7 %. Los diez peores casos de `eval/RESULTS.md`
se agrupan en cuatro causas.

## Resultados por plantilla (F1 medio por documento, modo `pdf`)

| Tipo | Plantilla | Docs | F1 | Diagnóstico |
|---|---|---|---|---|
| form | `nocolon` | 4 | 0,00 | Etiquetas sin `:` |
| invoice | `prose` | 8 | 0,18 | Campos dentro de oraciones |
| contract | `letter` | 9 | 0,44 | Carta en prosa; solo `parties`, `currency` y `governing_law` |
| receipt | `prose` | 8 | 0,50 | Fecha, total y método en prosa |
| report | `memo` | 8 | 0,86 | `Period: FY2024` sí; `fiscal year 2024` en prosa no |
| resto (kv, pos, block, es, schedule, recital, annual, technical, application, request) | 55 | 1,00 | En distribución de las reglas |

## Diez peores casos

| # | Caso | Causa | Hipótesis |
|---|---|---|---|
| 1–4 | `syn-frm-010`, `-011`, `-014`, `-018` (form `nocolon`) | El partidor de layout solo emite `KeyValue` cuando hay `:`; las líneas `Form ID   HR-1234` se clasifican como `Table` (dos espacios) y no producen pares. | Falta una regla de "etiqueta conocida + separación de ≥ 2 espacios". Es exactamente el patrón de FUNSD, donde la clave y el valor son cajas distintas. |
| 5–9 | `syn-inv-001`, `-002`, `-007`, `-011`, `-016` (invoice `prose`) | `Invoice INV-… was issued by … on March 14, 2026 to …` no tiene etiquetas. Solo se recupera `currency` (regex global) y un `total` **equivocado**: el regex de respaldo `total[^\d]{0,25}([\d.,]+)` captura el primer número tras la palabra *total* en "the amount … (subtotal 1,455.00 …" → toma el subtotal. | Los siete falsos positivos de `total` (P 78 %) vienen todos de aquí. El regex de respaldo debe exigir que la palabra sea *total* completa y no *subtotal*, o preferir el mayor de los montos candidatos. |
| 10 | `syn-inv-009` (invoice `prose`) | Igual que arriba, más `missing_required` en seis campos. | Mismo remedio; con llave, la ruta `vlm_fallback` (a la que el scorer envía estos documentos) es la que debe resolverlos. |

## Otras familias de error

- **Cartas de contrato** (`letter`, F1 0,44): "commencing on 14/03/2026 for a
  period of 24 months" no coincide con los patrones `as of …` ni
  `term … is …`. Añadir `commencing (?:on )?` y `period of (\d+) months`
  subiría el recall de `effective_date` y `term_months` (ambos 55 %).
- **Escaneos** (`scanned`, F1 81,0 % vs 82,9 % en `pdf`): tesseract lee
  `Form ID: HR-1234` como `Form ID: HR-1234.` o parte los guiones; diez
  documentos pierden un campo respecto al PDF (formularios `request` y
  reportes `technical`/`memo`, todos por `date`/`fiscal_year` con dígitos
  confundidos). Un post-procesado de OCR (mapa `O→0`, `l→1` en campos
  numéricos) y confianza por palabra en la penalización serían el siguiente
  paso.
- **Tablas sin líneas** (`syn-tbl-009`, F1 0): pdfplumber con estrategia
  `text` no detecta la tabla cuando las columnas están muy separadas y hay
  pocas filas; Camelot `stream` (extra `heavy`) sí la recupera en pruebas
  manuales.
- **FUNSD real** (F1 de pares 14 %, recall 8 %): las respuestas están en otra
  línea, en casillas o son múltiples por pregunta (`0001118259` tiene 23
  respuestas para 3 preguntas). La reconstrucción por líneas pierde la
  relación; hace falta usar las cajas (`box`) o el modelo de visión.
- **Clasificador**: 99,1 % de acierto; el único fallo es `funsd-0000989556`
  (un formulario con la palabra *report* varias veces) clasificado como
  `report`.

## Qué sí funciona

Precisión ≥ 94,8 % en todos los tipos: cuando las reglas extraen un valor,
casi siempre es correcto. El problema es recall, y el scorer lo refleja: los
33 documentos enviados a `vlm_fallback` son mayoritariamente las plantillas
en prosa y sin separador. Esa es la señal correcta para gastar tokens.
