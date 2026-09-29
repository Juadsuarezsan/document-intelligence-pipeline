"""Build and execute ``notebooks/demo.ipynb`` (no API key needed).

The notebook is generated from the cells below so it always matches the code,
then executed in place with ``nbclient`` so the committed file carries real
outputs. Requires the ``notebook`` extra: ``pip install -e ".[notebook]"``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "demo.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        "# Document Intelligence Pipeline: recorrido de punta a punta\n\n"
        "Este cuaderno ejecuta el pipeline **sin llave de API** (fallback determinista: "
        "pdfplumber + tesseract + reglas) sobre un texto, un PDF con capa de texto, un PDF "
        "escaneado y una anotación real de FUNSD, y muestra los campos, hallazgos, routing y "
        "telemetría. Con `ANTHROPIC_API_KEY` en `.env` los mismos pasos usan Claude y Claude Vision.",
    ),
    (
        "code",
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd().resolve()\n"
        "if not (ROOT / 'src').exists():\n"
        "    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT)); os.chdir(ROOT)\n"
        "os.environ.setdefault('LOG_LEVEL', 'WARNING')\n"
        "from src.config import Settings\n"
        "from src.observability.tracing import configure_logging\n"
        "configure_logging('WARNING')\n"
        "from src.pipeline.graph import DocumentPipeline, build_deps\n"
        "settings = Settings()\n"
        "pipeline = DocumentPipeline(build_deps(settings))\n"
        "print('modelo fijado:', settings.anthropic_model, '| llm_enabled:', settings.llm_enabled)",
    ),
    ("markdown", "## 1. Texto de una factura"),
    (
        "code",
        "INVOICE = '''ACME CONSULTING S.A.S.\nNIT: 900.123.456-8\nInvoice No: INV-2026-0042\n"
        "Invoice Date: 2026-03-14\nDue Date: 2026-04-13\nPayment Terms: Net 30\n"
        "Subtotal: 2,100.00\nTax (19%): 399.00\nTotal Due: USD 2,499.00'''\n"
        "result = await pipeline.run(text=INVOICE)\n"
        "print('tipo:', result.document_type, '| ruta:', result.pipeline_used, '| routing:', result.routing, '| conf:', result.overall_confidence)\n"
        "for f in result.fields:\n"
        "    print(f'  {f.name:15s} {str(f.value):22s} conf={f.confidence:.2f} src={f.source}')\n"
        "print('hallazgos:', [f.code for f in result.findings])",
    ),
    ("markdown", "## 2. PDF con capa de texto y tabla (pdfplumber)"),
    (
        "code",
        "from src.synth.pdf import render_document_pdf\n"
        "pdf = render_document_pdf(INVOICE, ['Description', 'Qty', 'Unit price', 'Amount'],\n"
        "                          [['Consulting hours', '10', '120.00', '1200.00'], ['Cloud hosting', '1', '900.00', '900.00']])\n"
        "result = await pipeline.run(pdf_bytes=pdf)\n"
        "print('ruta:', result.pipeline_used, '| páginas:', result.n_pages, '| tablas:', len(result.tables))\n"
        "if result.tables:\n"
        "    print(result.tables[0].headers); print(result.tables[0].rows)\n"
        "print('nodos:', [(n['node'], n['ms']) for n in result.node_log])",
    ),
    (
        "markdown",
        "## 3. PDF escaneado (rasterizado, sin capa de texto) → ruta OCR\n\nSi tesseract no está instalado el pipeline lo indica y usa la ruta de texto (vacía).",
    ),
    (
        "code",
        "from src.parsers.ocr_parser import tesseract_available\n"
        "from src.synth.pdf import render_scanned_pdf\n"
        "scanned = render_scanned_pdf(pdf)\n"
        "print('tesseract disponible:', tesseract_available(settings.tesseract_cmd))\n"
        "result = await pipeline.run(pdf_bytes=scanned, document_type='invoice')\n"
        "print('ruta:', result.pipeline_used, '| routing:', result.routing, '| latencia ms:', result.latency_ms)\n"
        "print({f.name: f.value for f in result.fields})",
    ),
    ("markdown", "## 4. Formulario real de FUNSD (texto reconstruido de la anotación)"),
    (
        "code",
        "from src.synth.funsd import load_funsd_annotation\n"
        "doc = load_funsd_annotation(ROOT / 'data/funsd_sample/annotations/0000971160.json')\n"
        "print(doc.text[:400]); print('...')\n"
        "print('pares anotados:', doc.pairs[:4])\n"
        "result = await pipeline.run(text=doc.text)\n"
        "print('tipo:', result.document_type, '| routing:', result.routing, '| campos:', {f.name: f.value for f in result.fields})",
    ),
    ("markdown", "## 5. Métricas de la última corrida (`eval/RESULTS.md`)"),
    (
        "code",
        "runs = sorted((ROOT / 'eval/runs').glob('*.json'))\n"
        "run = json.loads(runs[-1].read_text(encoding='utf-8'))\n"
        "summary = run['pipelines']['heuristic']['summary']\n"
        "print(runs[-1].name)\n"
        "print({k: summary[k] for k in ('documents', 'field_f1_micro', 'field_f1_macro', 'doc_accuracy', 'classifier_accuracy')})\n"
        "print('tablas cell F1:', run['tables']['f1'], '| latencia:', summary['latency'])\n"
        "print('pendiente con llave:', [k for k, v in run['pipelines'].items() if v.get('summary') is None])",
    ),
]


def build() -> nbformat.NotebookNode:
    """Create the notebook object from ``CELLS``."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    for kind, source in CELLS:
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(source))
    return nb


def main() -> int:
    """Build, execute and write the notebook."""
    nb = build()
    client = NotebookClient(
        nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    client.execute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT)
    n_out = sum(len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
    print(f"wrote {OUT} with {n_out} outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
