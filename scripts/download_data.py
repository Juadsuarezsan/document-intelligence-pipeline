"""Download the public datasets with checksum verification.

Usage::

    python scripts/download_data.py --dataset funsd            # ~17 MB zip
    python scripts/download_data.py --dataset pubtables1m-sample
    python scripts/download_data.py --all --dry-run             # print plan only

Datasets and licences:

* FUNSD (Jaume, Ekenel, Thiran, 2019) — https://guillaumejaume.github.io/FUNSD/
  199 scanned forms with entity/relation annotations. Non-commercial research
  licence; the archive is the official ``dataset.zip``.
* PubTables-1M (Smock, Pesala, Abraham, 2021) — https://github.com/microsoft/table-transformer
  Tables from PubMed Central with cell annotations. Community Data License
  Agreement – Permissive 1.0. The hosted archives are large; this script pulls
  a small sample listing so the table evaluation can be extended.
* DocVQA requires an account on the RRC portal and is NOT downloaded here.

The SHA-256 of every archive is recorded in ``data/MANIFEST.txt`` after a
successful download. A ``--expected-sha256`` mismatch aborts.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "MANIFEST.txt"


@dataclass(frozen=True)
class Dataset:
    """A downloadable archive."""

    name: str
    url: str
    filename: str
    licence: str
    extract: bool = True


DATASETS: dict[str, Dataset] = {
    "funsd": Dataset(
        name="funsd",
        url="https://guillaumejaume.github.io/FUNSD/dataset.zip",
        filename="funsd_dataset.zip",
        licence="FUNSD research licence (non-commercial)",
    ),
    "pubtables1m-sample": Dataset(
        name="pubtables1m-sample",
        url="https://raw.githubusercontent.com/microsoft/table-transformer/main/README.md",
        filename="pubtables1m_README.md",
        licence="CDLA-Permissive-1.0 (dataset); MIT (repository)",
        extract=False,
    ),
}


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file (streamed)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, client: httpx.Client, timeout: float = 120.0) -> Path:
    """Stream ``url`` into ``dest`` and return the path.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", url, follow_redirects=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)
    return dest


def record_manifest(name: str, path: Path, digest: str, manifest: Path = MANIFEST) -> None:
    """Append (or replace) the manifest line for ``name``."""
    line = f"{digest}  {path.relative_to(ROOT).as_posix()}  # {name}\n"
    existing = (
        manifest.read_text(encoding="utf-8").splitlines(keepends=True) if manifest.exists() else []
    )
    kept = [ln for ln in existing if not ln.rstrip().endswith(f"# {name}")]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(kept) + line, encoding="utf-8")


def fetch_dataset(
    ds: Dataset,
    *,
    client: httpx.Client,
    raw_dir: Path = RAW_DIR,
    expected_sha256: str | None = None,
    manifest: Path = MANIFEST,
) -> str:
    """Download, verify, extract and record one dataset; returns its SHA-256.

    Raises:
        ValueError: If ``expected_sha256`` is given and does not match.
    """
    dest = raw_dir / ds.filename
    download(ds.url, dest, client)
    digest = sha256_file(dest)
    if expected_sha256 and digest != expected_sha256:
        dest.unlink(missing_ok=True)
        raise ValueError(f"{ds.name}: sha256 mismatch (got {digest}, expected {expected_sha256})")
    if ds.extract and zipfile.is_zipfile(dest):
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(raw_dir / ds.name)
    record_manifest(ds.name, dest, digest, manifest)
    return digest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    names = sorted(DATASETS) if args.all else args.dataset
    if not names:
        parser.error("choose --dataset NAME or --all")
    for name in names:
        ds = DATASETS[name]
        if args.dry_run:
            print(f"[dry-run] {ds.name}: {ds.url} -> data/raw/{ds.filename} ({ds.licence})")
            continue
        with httpx.Client() as client:
            digest = fetch_dataset(ds, client=client, expected_sha256=args.expected_sha256)
        print(f"{ds.name}: sha256={digest} recorded in {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
