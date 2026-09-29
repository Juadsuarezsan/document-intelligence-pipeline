# Seguridad

## Secretos

- Ninguna llave en el repositorio. `.env` está en `.gitignore`; `.env.example`
  contiene solo nombres y comentarios.
- `gitleaks detect --no-banner --redact` (v8.21.2, binario de GitHub
  Releases) sobre los 26 commits de la rama `claude/fase-a` (historial completo):
  **no leaks found** (2026-09-29). La CI repite el escaneo con
  `gitleaks/gitleaks-action@v2` en cada push y PR.
- El cliente de Claude recibe la llave desde `Settings`; nunca se loguea. Las
  respuestas del modelo se registran solo como conteos de tokens.

## Validación de entrada (422)

- `ExtractRequest` exige exactamente uno de `text` o `pdf_b64` no vacíos
  (`model_validator`); tipos de documento fuera del `Literal` fallan en
  Pydantic.
- El PDF se decodifica con `validate=True`, se rechaza si supera
  `MAX_PDF_BYTES` (antes de decodificar, por longitud del base64) o si no
  empieza por `%PDF`; el texto se rechaza si supera `MAX_TEXT_CHARS`.
- `/api/extract/upload` aplica las mismas comprobaciones a los bytes subidos.
- Errores de pdfminer al abrir el archivo se traducen a 422, no a 500.

## Límite de tasa y CORS

- `slowapi` con `RATE_LIMIT` (por defecto `30/minute` por IP) en los dos
  endpoints de extracción; exceso → 429.
- CORS se lee de `CORS_ALLOW_ORIGINS`; el valor por defecto son orígenes
  locales y nunca `*`. Métodos `GET`/`POST`, cabeceras `Content-Type` y
  `X-Trace-Id`.

## PII

`src/validators/pii.py` enmascara correos, SSN, IBAN, teléfonos y tarjetas
(Luhn) en los valores extraídos antes de devolverlos, persistirlos o
loguearlos; los campos identificadores y montos están en `SAFE_FIELDS` para no
destruir números de factura. Activo por defecto (`REDACT_PII=true`) y cubierto
por `tests/test_validators.py::test_pii_redaction` y
`tests/test_pipeline.py::test_pii_is_redacted_and_reported`.

## Dependencias

Versiones fijadas con `==` en `pyproject.toml`. Las dependencias pesadas con
superficie nativa (`unstructured`, `camelot`/OpenCV) son opcionales.

## Pendiente

- HTTPS: lo aporta la plataforma de despliegue (sin despliegue aún).
- Autenticación del API: no implementada; el diseño asume un gateway
  delante en producción.
