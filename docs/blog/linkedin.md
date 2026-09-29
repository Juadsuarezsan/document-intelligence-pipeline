# Post para LinkedIn (borrador)

Publiqué el cuarto proyecto de mi portafolio de AI Engineering: un pipeline
de Document Intelligence que convierte facturas, recibos, contratos,
formularios y reportes en JSON validado, y que decide con honestidad si el
resultado se aprueba solo, va a revisión humana o se reprocesa con Claude
Vision.

Lo que más me importaba no era la extracción sino la fontanería alrededor:

- Parsing por capas: pdfplumber para PDFs con texto (18 ms), Tesseract para
  escaneos (0,7 s/página), Claude Vision para lo que el OCR no resuelve.
- Extracción guiada por esquema Pydantic y validación de negocio: campos
  requeridos, fechas ISO, subtotal + IVA = total y el dígito de verificación
  del NIT con el algoritmo de la DIAN.
- Confianza ponderada por cobertura y enrutamiento con umbrales (0,90 / 0,70),
  orquestado en LangGraph con trazas por nodo, tokens y costo por petición.
- Redacción de PII antes de persistir.

Medí sin gastar un token: 105 documentos (100 sintéticos con semilla,
declarados como tales, y 5 formularios reales de FUNSD), tres modos de entrada
(texto, PDF, escaneado por OCR) y 30 tablas. La línea base de reglas da Field
F1 82,9 % y Doc Accuracy 71 % sobre PDF, con precisión ≥ 95 % en todos los
tipos; el recall cae en prosa y en formularios reales, que es exactamente lo
que el scorer envía al modelo de visión. Las filas con Claude quedan
pendientes de llave; el código está probado con mocks.

Todo reproducible: `make install && make test && make eval`. 100+ tests,
cobertura 94 %, mypy --strict, CI con gitleaks.

Repo: https://github.com/Juadsuarezsan/document-intelligence-pipeline
Artículo con las decisiones y los errores que encontré: (enlace al post)

#AIEngineering #DocumentAI #IDP #LangGraph #Claude #Python #MLOps
