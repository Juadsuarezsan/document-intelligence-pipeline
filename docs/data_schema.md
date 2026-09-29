# Esquema de datos

## `data/eval/test_set.jsonl` (105 registros)

Un objeto JSON por línea, claves ordenadas (hash estable).

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | str | `syn-<tipo>-NNN` o `funsd-<id de anotación>` |
| `source` | `synthetic` \| `funsd` | Origen. Los sintéticos se generan con `scripts/generate_synthetic.py` (semilla `20260929`) |
| `doc_type` | `invoice` \| `receipt` \| `contract` \| `form` \| `report` | Tipo verdadero |
| `template` | str | Plantilla de layout (`kv_en`, `es`, `block`, `prose`, `pos`, `kv`, `recital`, `schedule`, `letter`, `application`, `request`, `nocolon`, `annual`, `technical`, `memo`, `funsd_annotation`) |
| `text` | str | Texto del documento. En FUNSD, reconstruido en orden de lectura a partir de las cajas (líneas separadas por `\n`, celdas por dos espacios) |
| `ground_truth` | objeto | Campos del esquema con valores normalizados (ver abajo). En FUNSD contiene solo `pairs` |
| `ground_truth.pairs` | lista de `[pregunta, respuesta]` | Pares enlazados `question → answer` (FUNSD) o campos extra del formulario (sintéticos) |
| `table` | objeto \| null | `{headers, rows, ruled}` de la tabla que se dibuja en la versión PDF (solo facturas) |
| `notes` | lista de str | `math_mismatch_injected` cuando el total se alteró a propósito; en FUNSD, conteo de entidades y etiquetas |

Valores del ground truth por campo:

| Campo | Tipo | Rango / formato |
|---|---|---|
| `invoice_number`, `receipt_number`, `contract_number`, `form_id` | str | Identificador tal como aparece |
| `vendor`, `submitter`, `author`, `title`, `department`, `governing_law`, `payment_terms`, `payment_method` | str | Texto; comparación normalizada (minúsculas, espacios, puntuación) y por contención para nombres |
| `vendor_tax_id` | str | NIT `NNN.NNN.NNN-D` con dígito de verificación válido |
| `issue_date`, `due_date`, `effective_date`, `date` | str | ISO-8601 `YYYY-MM-DD`, año 2026 |
| `currency` | str | ISO-4217 (`USD`, `EUR`, `COP`, `GBP`, `MXN`) |
| `subtotal`, `tax`, `total`, `total_value` | float | ≥ 0, dos decimales; `subtotal + tax = total` salvo `math_mismatch_injected` |
| `term_months` | int | 6, 12, 24 o 36 |
| `parties` | str | `"A; B"` |
| `fiscal_year` | str | `2024` o `2025` |

## `data/eval/tables.jsonl` (30 registros)

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | str | `syn-tbl-NNN` |
| `kind` | `items` \| `financial` \| `schedule` | Familia de tabla |
| `headers` | lista de str | Fila de encabezado |
| `rows` | lista de listas de str | Cuerpo (2–6 filas) |
| `ruled` | bool | `true` dibuja la rejilla completa; `false` solo una línea bajo el encabezado (2 de cada 3 son `ruled`) |

## `data/funsd_sample/annotations/*.json` (5 archivos)

Esquema original de FUNSD (Jaume et al., 2019):

| Campo | Tipo | Descripción |
|---|---|---|
| `form` | lista de entidades | |
| `form[].id` | int | Identificador dentro del formulario |
| `form[].text` | str | Texto de la entidad |
| `form[].box` | `[x0, y0, x1, y1]` | Caja en píxeles de la imagen original |
| `form[].label` | `header` \| `question` \| `answer` \| `other` | Etiqueta semántica |
| `form[].linking` | lista de `[id_origen, id_destino]` | Relaciones; se usan las `question → answer` |
| `form[].words` | lista de `{text, box}` | Palabras con caja |

Estadísticas de la muestra: 367 entidades, 281 pares pregunta/respuesta.
Conteo por archivo en `notes` de cada registro FUNSD del eval set.

## `data/MANIFEST.txt`

SHA-256 de los dos JSONL y de las cinco anotaciones. Regenerar con
`python scripts/build_manifest.py`; la CI falla si una generación fresca
difiere del manifiesto. `scripts/download_data.py` añade una línea por
archivo descargado.

## Salidas del evaluador

`eval/runs/<fecha>-<nombre>.json`: metadatos (`generated_at`, `git_commit`,
`model`, `llm_enabled`, `tesseract_available`), `dataset`, `pipelines.<nombre>`
con `modes.<text|pdf|scanned>` (métricas agregadas), `cases.<modo>` (detalle
por documento: `f1`, `missed`, `wrong`, `findings`, `predicted`), `summary`,
`worst_cases` y `tables` (métricas por tabla). `eval/RESULTS.md` se deriva de
este archivo.
