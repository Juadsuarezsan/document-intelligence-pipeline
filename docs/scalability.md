# Escalabilidad

## Capacidad actual (una réplica, sin llave)

Medido en la corrida `eval/runs/2026-09-29-heuristic-baseline.json` sobre un
contenedor sin GPU:

| Ruta | Latencia media | p95 | Cuello de botella |
|---|---|---|---|
| Texto ya extraído | 5 ms | 7 ms | CPU (regex) |
| PDF con capa de texto | 18 ms | 41 ms | pdfminer (pdfplumber) |
| PDF escaneado (OCR) | 688 ms | 1 480 ms | tesseract, una página a 150 dpi |
| Tablas (pdfplumber) | 8 ms | — | pdfminer |

Con llave, cada llamada a Claude añade del orden de 2–6 s (clasificación +
extracción; la visión con imágenes de 150 dpi es más lenta) y el costo por
documento pasa de 0 a un rango estimado de 0,01–0,05 USD según páginas y
reprocesamiento (precio de lista 3 USD / 15 USD por millón de tokens de
entrada / salida; no medido aún, ver `eval/RESULTS.md`).

## Estimación de costo a 1 000 usuarios mensuales

Supuestos explícitos: 1 000 usuarios × 30 documentos/mes = 30 000
documentos; 1,5 páginas por documento; 33 % de los documentos llegan a
`vlm_fallback` (fracción observada con las reglas); una imagen de página
≈ 1 500 tokens de entrada; extracción ≈ 2 000 tokens de entrada y 300 de
salida; clasificación ≈ 700 de entrada y 30 de salida.

| Partida | Cálculo | USD/mes |
|---|---|---|
| Clasificación + extracción con Claude (todos los docs) | 30 000 × (2 700 in × 3 + 330 out × 15) / 1e6 | ≈ 392 |
| Reprocesamiento con visión (33 %) | 10 000 × (1,5 × 1 500 + 500) in × 3 / 1e6 + salida | ≈ 90 |
| API (2 vCPU, 4 GB) + Postgres gestionado | Railway/Fly + Neon/RDS pequeño | ≈ 60–120 |
| **Total** | | **≈ 550–600 USD/mes** (≈ 0,02 USD/documento) |

Si solo se usa Claude cuando las reglas no aprueban (modo *fallback* del
spec) el primer renglón cae al 33 % de los documentos: ≈ 130 USD y un total
cercano a 300 USD/mes. Es el argumento económico de tener reglas primero.

## 100× (3 M documentos/mes, ~70 docs/min sostenidos)

- El API es `async`; el OCR es CPU-bound y bloquea el event loop. Primer
  cambio: ejecutar tesseract en un `ProcessPoolExecutor` o en un worker
  separado (cola Redis/SQS) con 4–8 procesos por nodo.
- Claude: ~2 300 llamadas/hora; entra en los límites de tasa por defecto de
  Anthropic. Solicitar tier superior y usar Batches API para el reprocesamiento
  no interactivo (50 % de descuento, latencia de horas).
- Postgres: la tabla `processed_documents` con JSONB crece ~10 KB/doc → 30 GB
  al año; particionar por mes y mover `result` a almacenamiento de objetos.
- Prompt caching sobre el prompt de sistema del extractor (estable por tipo
  de documento) reduce el costo de entrada ~90 % en el prefijo.

## 1 000× (30 M documentos/mes)

- Separar en servicios: ingest/OCR (CPU, autoscaling por cola), extracción
  LLM (limitada por cuota), validación/scoring (barato, junto al API).
- Idempotencia por hash del PDF para no reprocesar duplicados (frecuente en
  cuentas por pagar).
- Cola de revisión humana como servicio propio con SLA; el 60 % de los
  documentos que hoy van a `human_review` necesitan una UI de corrección
  cuyos datos alimenten un fine-tune del clasificador y nuevas reglas.
- Modelo más barato (Haiku) para clasificación; Sonnet solo para extracción y
  visión.

## Lo que no escala y hay que rehacer

- El extractor heurístico por líneas: en formularios reales (FUNSD) el recall
  de pares es 8 %. La solución no es más regex sino extracción por
  proximidad geométrica (cajas OCR) o el modelo de visión.
- `RecentRequests` en memoria: sirve para un dashboard local; en producción
  el histórico vive en Postgres/LangSmith.
- Rate limiting por IP en memoria (`slowapi`): con varias réplicas se
  necesita el backend Redis de `limits`.
