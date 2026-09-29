# Decisiones técnicas

Cada entrada registra la elección, la alternativa descartada y el criterio.

## 1. Extractor heurístico real como línea base y primer paso

**Elección:** `HeuristicExtractor` con variantes de etiqueta por campo,
normalización de montos/fechas/moneda, reglas en prosa para contratos y
partición de líneas multi-celda.
**Alternativa:** un *stub* que devuelva vacío sin llave y delegar todo en el
modelo.
**Criterio:** sin una línea base medible la tabla comparativa del spec no
tiene primera fila y no se puede cuantificar cuánto aporta Claude. Además
reduce el costo: el 7 % de los documentos sintéticos se aprueba solo y el
62 % va a revisión humana sin gastar un token.

## 2. pdfplumber para la ruta de texto; `unstructured` y Camelot opcionales

**Elección:** `pdfplumber` (texto + tablas con líneas) como dependencia base;
`unstructured` y `camelot-py` en el extra `heavy`, importados de forma
perezosa y con fallback.
**Alternativa:** `unstructured.partition.pdf` o `docling` como parser único.
**Criterio:** `unstructured` 0.16 descarga modelos NLTK en el primer uso
(falla en redes restringidas, verificado) y su ruta de PDF arrastra
`unstructured-inference` con `torch`; `docling` también depende de `torch`.
pdfplumber cubre el 100 % de las tablas con líneas y el 91 % sin líneas en la
evaluación, sin dependencias nativas.

## 3. Tesseract 5 + pypdfium2 para escaneos

**Elección:** rasterizar con `pypdfium2` (ya viene con pdfplumber) a 150 dpi
y leer con `pytesseract.image_to_data` (`--psm 6`).
**Alternativa:** PaddleOCR.
**Criterio:** instalación por `apt` sin GPU, confianza por palabra que
alimenta el hallazgo `low_ocr_confidence`, y latencia media de 0,7 s por
página. PaddleOCR mejora escaneos degradados pero requiere PaddlePaddle
(>500 MB) y no aportaba a la evaluación disponible.

## 4. SDK `anthropic` directo, no `langchain-anthropic`

**Elección:** `ClaudeClient` propio sobre `anthropic.AsyncAnthropic` con
`max_retries=0`, timeout explícito y `tenacity` (backoff exponencial solo en
429/5xx/conexión/timeout; 4xx no se reintenta). Bloques `image` en base64
para visión.
**Alternativa:** `ChatAnthropic` de LangChain.
**Criterio:** control de reintentos y de accounting (tokens y costo por
llamada), tipado estricto sin `SecretStr` ni argumentos no reconocidos por
mypy, y una sola dependencia de red que simular en tests.

## 5. LangGraph solo para orquestar

**Elección:** `StateGraph` con seis nodos y una arista condicional para el
reprocesamiento por visión.
**Alternativa:** una función `async` lineal.
**Criterio:** la lógica de reintento con visión es una bifurcación con
límite (una sola vez) que conviene ver en el grafo; el `TypedDict` de estado
documenta qué produce cada nodo; y LangSmith instrumenta el grafo sin código
adicional. El costo es una dependencia más y la convención `n_*` para evitar
colisiones nombre/clave.

## 6. Confianza ponderada por cobertura

**Elección:** `conf = media(conf_campo) × (0.5 + 0.5·cobertura_requeridos)
− 0.10·errores − 0.03·avisos`, con umbrales por variable de entorno.
**Alternativa:** media simple de confianzas.
**Criterio:** con la media simple una factura con solo `currency=USD` al 0.95
se aprobaba sola. Con la fórmula actual el 33 % de los documentos sintéticos
va a `vlm_fallback`, que es exactamente donde las reglas fallan (prosa,
cartas, formularios sin separador).

## 7. Validación del NIT con el algoritmo de la DIAN

**Elección:** dígito de verificación módulo 11 con los pesos oficiales
(3, 7, 13, 17, 19, 23, 29, 37, 41, 43, 47, 53, 59, 67, 71).
**Alternativa:** regex de forma `\d{9}-\d`.
**Criterio:** un regex acepta cualquier dígito; el algoritmo detecta errores
de OCR en el identificador fiscal, que es un campo crítico para cuentas por
pagar. El generador sintético produce NITs válidos para que la regla sea
medible.

## 8. Redacción de PII determinista

**Elección:** patrones para correo, SSN, IBAN, teléfono y tarjeta (esta
última con Luhn) aplicados a los valores extraídos antes de persistir y
loguear; identificadores y montos están en una lista segura.
**Alternativa:** Presidio.
**Criterio:** Presidio arrastra spaCy y modelos; los patrones cubren los
formularios médicos y de RR. HH. del caso de uso, y Luhn evita enmascarar
números de factura de 13–16 dígitos.

## 9. Datos sintéticos con semilla en lugar de omitir la evaluación

**Elección:** 100 documentos de cinco tipos con 15 plantillas, incluidas
variantes en prosa, carta, memo y formulario sin dos puntos, más 30 tablas;
ground truth normalizado; `source: synthetic` en cada registro.
**Alternativa:** esperar a FUNSD completo y DocVQA.
**Criterio:** la red del entorno bloqueaba las descargas. El conjunto
sintético permite medir la línea base, el clasificador, la ruta OCR y las
tablas de forma reproducible (manifiesto SHA-256 verificado en CI), y las 5
anotaciones FUNSD reales dan la referencia de dificultad real.

## 10. Un gate de cobertura del 70 % con 94 % medido

Se fija el mínimo del DoD en CI, no el máximo alcanzado, para que añadir
código exploratorio (nuevas plantillas, backends de tablas) no rompa la
integración; la cobertura real se reporta en el README.
