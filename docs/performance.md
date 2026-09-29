# Rendimiento

Fuente: `eval/runs/2026-09-29-heuristic-baseline.json` (105 documentos, sin
llave, contenedor de 2 vCPU sin GPU) y `node_log` de las peticiones.

## Dónde se va el tiempo

| Modo | Media | p50 | p95 | Nodo dominante |
|---|---|---|---|---|
| `text` | 5,3 ms | 5 ms | 7 ms | `n_extract` (partición + regex) |
| `pdf` | 17,8 ms | 13 ms | 41 ms | `n_ingest` (pdfminer abre y tokeniza la página) y `n_tables` (segunda apertura del PDF) |
| `scanned` | 688 ms | 416 ms | 1 480 ms | `n_ingest`: rasterizar a 150 dpi (~60 ms) + tesseract `--psm 6` (~350–1 300 ms según densidad) |
| Tablas sueltas | 8 ms | — | 10 ms | pdfplumber `extract_tables` |

Perfil de la ruta escaneada (una página, `image_to_data`): el tiempo de
tesseract crece con el número de palabras (los reportes con párrafos tardan
más que los recibos) y la resolución. A 100 dpi la latencia baja ~40 % pero el
recall de identificadores con guiones cae; 150 dpi fue el punto elegido.

## Cuello de botella principal y por qué

**Tesseract es síncrono y CPU-bound.** Se ejecuta dentro de un nodo `async`
del grafo, así que bloquea el event loop del API mientras dura: con varias
peticiones escaneadas concurrentes el p95 se degrada linealmente. Es
aceptable para la demo y el eval; en producción hay que moverlo a un
`ProcessPoolExecutor` (`asyncio.to_thread` no basta porque el GIL se libera
solo parcialmente en pytesseract, que lanza un subproceso) o a un worker.

Segundo cuello, cuando hay llave: las llamadas a Claude (2–6 s cada una,
hasta cuatro por documento en la ruta de visión). Mitigaciones ya
soportadas por el diseño: prompt de sistema estable por tipo (cacheable),
reprocesamiento solo por debajo del umbral, y Batches API para colas no
interactivas.

## Optimizaciones aplicadas

- Las páginas solo se rasterizan cuando la ruta es `ocr`/`vlm` o hay llave
  (para poder reprocesar); en modo texto sin llave no se genera ninguna
  imagen.
- Imports perezosos de pdfplumber, pypdfium2, pytesseract, camelot,
  unstructured y psycopg_pool: el arranque del API y los tests de dominio no
  pagan su costo.
- El extractor heurístico compila los regex a nivel de módulo y construye el
  índice clave→valor una sola vez por documento.
- `RecentRequests` es un `deque(maxlen=100)`; los percentiles se calculan
  sobre 100 enteros.

## Caching

- `get_settings()` está memoizado (`lru_cache`).
- No hay caché de resultados por documento todavía; el hash SHA-256 del PDF
  como clave de idempotencia es el siguiente paso (ver `scalability.md`).

## Batch

`python -m eval.run` procesa los 105 documentos secuencialmente (≈ 1 min 40 s
con OCR incluido). La evaluación es batch por diseño; el reporte de tablas
renderiza 30 PDFs y los vuelve a leer en menos de un segundo.
