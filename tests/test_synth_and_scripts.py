import hashlib
import io
import sys
import zipfile
from pathlib import Path

import httpx
import pytest
import respx

from src.eval.dataset import ROOT
from src.synth.documents import SEED, generate_documents, generate_tables, valid_nit
from src.synth.funsd import load_funsd_annotation, load_funsd_dir
from src.synth.pdf import render_document_pdf, render_scanned_pdf, render_table_pdf, render_text_pdf
from src.validators.regex_rules import is_valid_nit

SCRIPTS = ROOT / "scripts"


def _load_script(name: str):  # type: ignore[no-untyped-def]
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_generation_is_deterministic() -> None:
    a = [d.to_record() for d in generate_documents(3, SEED)]
    b = [d.to_record() for d in generate_documents(3, SEED)]
    assert a == b and len(a) == 15
    assert {d["doc_type"] for d in a} == {"invoice", "receipt", "contract", "form", "report"}
    assert all(d["source"] == "synthetic" for d in a)
    assert generate_tables(2, 1) == generate_tables(2, 1)
    import random

    assert is_valid_nit(valid_nit(random.Random(1)))


def test_synthetic_ground_truth_is_consistent_with_text() -> None:
    for doc in generate_documents(4, SEED):
        if doc.doc_type == "invoice" and "math_mismatch_injected" not in doc.notes:
            gt = doc.ground_truth
            assert round(gt["subtotal"] + gt["tax"], 2) == gt["total"]
        assert doc.ground_truth
        assert doc.text.strip()


def test_funsd_loader_builds_pairs_and_reading_order() -> None:
    docs = load_funsd_dir(ROOT / "data" / "funsd_sample" / "annotations")
    assert len(docs) == 5
    total_pairs = sum(len(d.pairs) for d in docs)
    assert total_pairs > 200
    one = load_funsd_annotation(ROOT / "data" / "funsd_sample" / "annotations" / "0000971160.json")
    assert one.n_entities == 24 and len(one.pairs) == 8
    assert "Date:" in one.text and one.to_record()["source"] == "funsd"


def test_pdf_renderers_produce_valid_pdfs() -> None:
    assert render_text_pdf("a\nb").startswith(b"%PDF")
    assert render_table_pdf(["h"], [["1"]], ruled=False).startswith(b"%PDF")
    doc = render_document_pdf("x", ["h"], [["1"]], ruled=True)
    assert doc.startswith(b"%PDF")
    assert render_scanned_pdf(doc, dpi=50).startswith(b"%PDF")


@respx.mock
def test_download_data_verifies_sha_and_records_manifest(tmp_path: Path) -> None:
    dl = _load_script("download_data")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("dataset/testing_data/annotations/1.json", '{"form": []}')
    payload = buf.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    respx.get(dl.DATASETS["funsd"].url).mock(return_value=httpx.Response(200, content=payload))
    manifest = tmp_path / "MANIFEST.txt"
    raw = tmp_path / "raw"
    with httpx.Client() as client:
        got = dl.fetch_dataset(
            dl.DATASETS["funsd"],
            client=client,
            raw_dir=raw,
            expected_sha256=digest,
            manifest=manifest,
        )
    assert got == digest
    assert (raw / "funsd" / "dataset" / "testing_data" / "annotations" / "1.json").exists()
    assert digest in manifest.read_text(encoding="utf-8")
    with httpx.Client() as client, pytest.raises(ValueError):
        dl.fetch_dataset(
            dl.DATASETS["funsd"],
            client=client,
            raw_dir=raw,
            expected_sha256="0" * 64,
            manifest=manifest,
        )


@respx.mock
def test_download_data_http_error_propagates(tmp_path: Path) -> None:
    dl = _load_script("download_data")
    respx.get(dl.DATASETS["funsd"].url).mock(return_value=httpx.Response(403))
    with httpx.Client() as client, pytest.raises(httpx.HTTPStatusError):
        dl.fetch_dataset(
            dl.DATASETS["funsd"], client=client, raw_dir=tmp_path, manifest=tmp_path / "m.txt"
        )


def test_download_data_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    dl = _load_script("download_data")
    assert dl.main(["--all", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[dry-run] funsd" in out and "guillaumejaume" in out


def test_build_manifest_matches_current_files() -> None:
    bm = _load_script("build_manifest")
    text = bm.build_manifest()
    assert "data/eval/test_set.jsonl" in text
    committed = (ROOT / "data" / "MANIFEST.txt").read_text(encoding="utf-8")
    assert committed == text, "data/MANIFEST.txt is stale: run python scripts/build_manifest.py"
