FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TESSERACT_CMD=/usr/bin/tesseract

WORKDIR /app

# tesseract-ocr enables the scanned-document path; libpq for psycopg; curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --upgrade pip && pip install -e .

COPY eval /app/eval
COPY data /app/data
COPY demo /app/demo
COPY scripts /app/scripts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
