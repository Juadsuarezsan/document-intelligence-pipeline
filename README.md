# Document Intelligence Pipeline

[![CI](https://github.com/Juadsuarezsan/document-intelligence-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Juadsuarezsan/document-intelligence-pipeline/actions/workflows/ci.yml)
[![Coverage gate](https://img.shields.io/badge/coverage-%E2%89%A570%25%20gate%20(94%25%20medido)-brightgreen)](#tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776ab.svg)](.python-version)
[![Model](https://img.shields.io/badge/Claude-claude--sonnet--4--5--20250929-7c5cff)](src/config.py)
[![Demo](https://img.shields.io/badge/demo-galer%C3%ADa%20est%C3%A1tica-22d3ee)](demo/index.html)

Procesamiento inteligente de documentos (IDP): recibe un PDF o texto, decide si
puede leerlo por su capa de texto, por OCR o con Claude Vision, extrae los
campos de un esquema Pydantic por tipo de documento, los valida con reglas
(NIT con dígito de verificación, fechas ISO, `subtotal + impuesto = total`),
calcula una confianza por campo y por documento y decide si el resultado se
aprueba solo, va a revisión humana o se reprocesa con el modelo de visión.

**Demo:** la galería estática (`demo/index.html`, 12 documentos pre-procesados)
funciona sin backend; el panel "en vivo" llama a `POST /api/extract` de un
API local. El despliegue público con URL persistente está pendiente (ver
[Limitaciones](#limitaciones-conocidas)).

## Qué hace este proyecto

Convierte facturas, recibos, contratos, formularios y reportes en JSON
estructurado y verificado. Cada campo llega con su valor normalizado
(fechas ISO, montos como números), una confianza y su origen (`regex`,
`extractor`, `vlm`), y cada documento con una decisión de enrutamiento:
`auto_approve`, `human_review` o `vlm_fallback`. Sin llave de API todo el
sistema funciona en modo determinista (pdfplumber + tesseract + reglas); con
`ANTHROPIC_API_KEY` se activan el clasificador, el extractor, la transcripción
por visión y el juez LLM.

## Caso de uso industrial

Es el problema central de legal-tech, fintech, seguros y salud: cuentas por
pagar que reciben miles de facturas de proveedores con formatos distintos,
aseguradoras que digitalizan formularios de reclamación escaneados, firmas de
abogados que extraen fechas y montos de contratos, hospitales que capturan
formularios de admisión con datos personales que hay que redactar antes de
almacenarlos. Empresas como Hyperscience, Rossum o Klarity venden exactamente
esta cadena: parsing por capas según el tipo de entrada, extracción guiada por
esquema, validación cruzada y una cola de revisión humana alimentada por la
confianza del sistema, no por muestreo ciego.

## Arquitectura

![Arquitectura del pipeline](docs/architecture.svg)

La tubería es un `StateGraph` de LangGraph con seis nodos y una arista
condicional (`src/pipeline/graph.py`):

| Nodo | Qué hace | Componente |
|---|---|---|
| `n_ingest` | Valida el PDF, mide la densidad de texto por página y elige la ruta `text` (pdfplumber), `ocr` (pypdfium2 + tesseract) o `vlm` (Claude Vision) | `src/parsers/` |
| `n_classify` | Tipo de documento por heurística ponderada, por Claude sobre texto, o por Claude Vision sobre la primera página cuando no hay texto | `src/classifier/` |
| `n_tables` | Extrae tablas con pdfplumber (líneas y luego alineación de texto); Camelot *stream* opcional | `src/tables/` |
| `n_extract` | Rellena el esquema del tipo con reglas (`HeuristicExtractor`) o con Claude (`LLMExtractor`, JSON validado) | `src/extractors/` |
| `n_validate` | Campos requeridos, NIT (módulo 11 DIAN), fechas, montos, aritmética, orden de fechas, plazo del contrato, redacción de PII | `src/validators/` |
| `n_score` | Confianza = media por campo × cobertura de requeridos − penalizaciones; umbrales 0.90 / 0.70 | `src/scoring/` |
| `n_vlm_reprocess` | Solo si `routing == vlm_fallback` y hay llave: extrae de nuevo desde las imágenes con Claude Vision y vuelve a validar | `src/extractors/llm_extractor.py::VLMExtractor` |

Cada nodo registra entrada, salida y duración bajo el `trace_id` de la
petición; la respuesta HTTP devuelve `node_log`, `latency_ms`, tokens y
`cost_usd`. Más detalle en [`docs/architecture.md`](docs/architecture.md).

## Métricas y resultados

Los números provienen exclusivamente de `eval/runs/*.json` generados por
`python -m eval.run`, que también reescribe [`eval/RESULTS.md`](eval/RESULTS.md)
y el bloque siguiente. **Solo la fila heurística está medida en este
repositorio**: es el fallback determinista sin LLM y no representa la calidad
del sistema completo. Las dos filas con Claude Vision requieren
`ANTHROPIC_API_KEY`; su código está implementado y probado con mocks.

<!-- METRICS:START -->
Última corrida: `eval/runs/2026-09-29-heuristic-baseline.json` (2026-09-29T01:39:02+00:00). Solo la fila heurística está medida (fallback determinista, sin LLM); las de Claude Vision requieren `ANTHROPIC_API_KEY`.

| Pipeline | Field F1 | Doc Accuracy | Table F1 | Latencia | Costo/doc |
|---|---|---|---|---|---|
| Unstructured only (heurístico: pdfplumber/OCR + reglas, sin LLM) | 82.9 % | 71.0 % | 96.7 % | 18 ms (p95 41 ms) | $0.0000 |
| Unstructured + Claude Vision fallback (este) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) |
| Claude Vision directo | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) |
<!-- METRICS:END -->

Notas de lectura:

- El conjunto de evaluación tiene 105 documentos: 100 sintéticos generados
  con semilla fija (`scripts/generate_synthetic.py`, 15 plantillas que
  incluyen variantes en prosa, cartas, memos y formularios sin dos puntos,
  diseñadas para que las reglas fallen donde un modelo debería ganar) y las 5
  anotaciones reales de FUNSD incluidas en `data/funsd_sample/`. Están
  marcados por `source` en `data/eval/test_set.jsonl`.
- Field F1 se calcula por tipo de campo (precisión y recall por nombre de
  campo, micro y macro); Doc Accuracy exige todos los campos críticos
  correctos; Table F1 es F1 a nivel de celda sobre 30 tablas sintéticas en
  PDF (20 con líneas, 10 sin líneas).
- La fila heurística se reporta sobre PDFs reales (modo `pdf`). El mismo
  conjunto se evalúa también como texto ya extraído y como PDF escaneado
  (rasterizado y leído con tesseract); las tres cifras están en `RESULTS.md`.
- En FUNSD el baseline solo obtiene pares `pregunta: respuesta` cuando están en
  la misma línea del texto reconstruido: F1 estricto de pares 14 % (P 78 %,
  R 8 %). Es la brecha que el reprocesamiento con Claude Vision debe cerrar.
- Los diez peores casos y sus causas están en
  [`docs/error_analysis.md`](docs/error_analysis.md).

## Quickstart

```bash
git clone https://github.com/Juadsuarezsan/document-intelligence-pipeline.git
cd document-intelligence-pipeline
make install                       # venv + pip install -e ".[dev]"
sudo apt-get install -y tesseract-ocr   # opcional: habilita la ruta OCR
make test lint                     # pytest --cov (gate 70 %), ruff, black, mypy --strict
make eval                          # regenera eval/runs/*.json, eval/RESULTS.md y este README
make run                           # API en http://localhost:8000 (docs en /docs)
```

Probar el API:

```bash
curl -s -X POST http://localhost:8000/api/extract \
  -H 'Content-Type: application/json' \
  -d '{"text": "ACME S.A.S.\nNIT: 900.123.456-8\nInvoice No: INV-1\nDate: 2026-03-14\nSubtotal: 100.00\nTax: 19.00\nTotal Due: USD 119.00"}' | python -m json.tool

curl -s -F file=@factura.pdf http://localhost:8000/api/extract/upload | python -m json.tool
```

Para la demo, abra `demo/index.html` (o `python -m http.server 8080 -d demo`)
y apunte el panel en vivo a `http://localhost:8000`. Con Docker:
`docker compose up --build` levanta Postgres, el API y la demo (pendiente de
validar: no hubo demonio Docker en el entorno de desarrollo).

Copie `.env.example` a `.env` para configurar llave, umbrales, CORS, límite
de tasa, tamaño máximo y LangSmith; cada variable está comentada.

## Decisiones técnicas

1. **Reglas primero, modelo después.** El extractor heurístico no es un
   *stub*: reconoce variantes de etiqueta, normaliza montos en formato
   `1.234,56` y `1,234.56`, fechas en cinco formatos y cláusulas en prosa de
   contratos. Es la línea base medible sin llave y el primer paso del
   pipeline real; Claude solo entra cuando la confianza cae. Esto reduce el
   costo por documento y hace reproducible la evaluación.
2. **pdfplumber en lugar de `unstructured` para la ruta de texto.**
   `unstructured` 0.16 necesita descargar modelos NLTK en tiempo de ejecución
   (bloqueado en entornos sin red) y su partición de PDF arrastra `torch`.
   pdfplumber da texto y tablas con ruling lines sin dependencias nativas;
   `unstructured` y Camelot quedan como extra opcional `heavy` con import
   perezoso.
3. **Tesseract vía pypdfium2 en vez de PaddleOCR.** Tesseract 5 se instala con
   `apt`, no necesita GPU y `image_to_data` devuelve confianza por palabra,
   que alimenta el hallazgo `low_ocr_confidence`. PaddleOCR mejora recall en
   escaneos difíciles pero requiere PaddlePaddle (>500 MB).
4. **SDK `anthropic` directo con tenacity, no LangChain.** Un cliente propio
   (`src/llm/client.py`) fija el modelo `claude-sonnet-4-5-20250929`, el
   timeout, desactiva los reintentos del SDK y aplica backoff exponencial solo
   en 429/5xx/conexión; cada llamada devuelve tokens y costo. LangGraph se usa
   únicamente para la orquestación, donde sus aristas condicionales hacen
   visible la lógica de reprocesamiento.
5. **Confianza como cobertura ponderada, no promedio simple.** Un documento
   con dos campos al 0.95 y seis requeridos ausentes no debe aprobarse solo;
   la confianza multiplica la media por `0.5 + 0.5·cobertura` y resta 0.10 por
   error de validación. Los umbrales 0.90/0.70 son variables de entorno.
6. **Validación con semántica local.** El NIT colombiano se verifica con el
   algoritmo módulo 11 de la DIAN (pesos 3, 7, 13, 17, …), no con un regex de
   longitud; los datos sintéticos se generan con dígitos válidos para que la
   regla sea medible.
7. **Redacción de PII antes de persistir.** Correos, SSN, IBAN, teléfonos y
   tarjetas (con Luhn, para no enmascarar números de factura) se enmascaran en
   los valores extraídos y se reporta el conteo como hallazgo `pii_redacted`.

Más alternativas descartadas en [`docs/decisions.md`](docs/decisions.md).

## Tests

```bash
make test   # 100+ tests, cobertura medida 94 % con gate en 70 %
```

Incluye mocks del SDK de Anthropic (`FakeAnthropic` en `tests/conftest.py`),
pruebas de reintentos (429 reintenta, 400 no), casos límite del API (vacío,
sobredimensionado, base64 malformado, no-PDF → 422; límite de tasa → 429),
la ruta OCR con `pytesseract` simulado y con tesseract real cuando está
instalado, la ruta Vision forzada y el reprocesamiento por baja confianza con
respuestas enlatadas, y una corrida corta del evaluador.

## Observabilidad

Cada petición lleva `X-Trace-Id`; el log (loguru) incluye el `trace_id`, y
la respuesta devuelve `latency_ms`, `usage.input_tokens`,
`usage.output_tokens`, `usage.cost_usd` y el `node_log` con la duración de
cada nodo. `GET /api/metrics/recent` expone las últimas 100 peticiones con
p50/p95 y costo acumulado. Con `LANGSMITH_API_KEY` y
`LANGCHAIN_TRACING_V2=true` LangGraph envía trazas a LangSmith sin cambios de
código (pendiente de ejecutar: requiere llaves).

## Datos

| Fuente | Uso | Licencia | Cómo obtenerla |
|---|---|---|---|
| [FUNSD](https://guillaumejaume.github.io/FUNSD/) (199 formularios escaneados) | Pares pregunta/respuesta reales; 5 anotaciones incluidas en `data/funsd_sample/` | Licencia de investigación FUNSD (no comercial) | `python scripts/download_data.py --dataset funsd` |
| Sintéticos (`src/synth/`) | 100 documentos + 30 tablas con verdad de terreno, semilla `20260929` | MIT (este repo) | `make data` |
| [PubTables-1M](https://github.com/microsoft/table-transformer) / FinTabNet | Extensión de la evaluación de tablas | CDLA-Permissive-1.0 | `scripts/download_data.py --dataset pubtables1m-sample` |
| [DocVQA](https://www.docvqa.org/) | Document-level accuracy (pendiente) | Requiere cuenta en el portal RRC | Descarga manual |

El esquema de cada archivo está en [`docs/data_schema.md`](docs/data_schema.md)
y los SHA-256 en `data/MANIFEST.txt` (la CI verifica que una generación
fresca coincide con el manifiesto).

## Limitaciones conocidas

- **Sin corridas con Claude en el repositorio.** Las filas *Claude Vision
  fallback* y *Claude Vision directo* de la tabla, el juez LLM y las trazas
  en LangSmith requieren `ANTHROPIC_API_KEY`; hoy son código probado con mocks.
- **El eval sintético mide las reglas en su distribución.** Las plantillas
  clave-valor se resuelven al 100 %; las de prosa, cartas y formularios sin
  separador caen a 0–50 %. Los 5 formularios FUNSD reales muestran la brecha
  verdadera (recall de pares 8 %). El conjunto FUNSD completo y DocVQA no se
  pudieron descargar en el entorno de desarrollo.
- **La extracción de tablas es posicional.** pdfplumber acierta 100 % con
  líneas y 91 % sin líneas; una tabla sin líneas con columnas muy separadas
  no se detecta (`syn-tbl-009`). No hay Table Transformer para tablas en
  imagen.
- **OCR a 150 dpi con `--psm 6`.** Suficiente para impresos; en escaneos
  inclinados o manuscritos el recall cae y el sistema depende del
  reprocesamiento por visión.
- **Sin despliegue público ni Docker validado** en el entorno de desarrollo.

## Trabajo futuro

1. Ejecutar `make eval` con `ANTHROPIC_API_KEY` para completar la tabla
   obligatoria y publicar 30 trazas en LangSmith.
2. Descargar FUNSD completo (50 formularios de test) y reportar el F1 de
   pares sobre el conjunto oficial; añadir DocVQA para document-level accuracy.
3. Extracción de pares por proximidad geométrica (cajas de FUNSD / OCR) en
   lugar de solo por línea, que es la causa principal del recall bajo.
4. Table Transformer (TATR) para tablas en imagen y Camelot *lattice* con
   Ghostscript para PDFs con celdas combinadas.
5. Despliegue del API (Railway/HF Spaces) con p95 medido y la galería en
   GitHub Pages.

## Autor

Juan David Suárez Sánchez · juadsuarezsan@unal.edu.co ·
[LinkedIn](https://www.linkedin.com/in/juan-david-suarez-sanchez-31ab281b7)

Licencia MIT (ver [`LICENSE`](LICENSE)).
