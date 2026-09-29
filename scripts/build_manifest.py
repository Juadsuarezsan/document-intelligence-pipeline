"""Write SHA-256 digests of the versioned data files to ``data/MANIFEST.txt``.

Run after ``scripts/generate_synthetic.py``; the CI compares the manifest with
a fresh generation to prove the evaluation set is reproducible.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "MANIFEST.txt"
TRACKED = [
    "data/eval/test_set.jsonl",
    "data/eval/tables.jsonl",
    "data/funsd_sample/annotations/0000971160.json",
    "data/funsd_sample/annotations/0000989556.json",
    "data/funsd_sample/annotations/0000990274.json",
    "data/funsd_sample/annotations/0000999294.json",
    "data/funsd_sample/annotations/0001118259.json",
]


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(root: Path = ROOT, tracked: list[str] = TRACKED) -> str:
    """Return the manifest text (one ``<sha256>  <path>`` line per file)."""
    lines = ["# SHA-256 of versioned data files. Regenerate: python scripts/build_manifest.py"]
    for rel in tracked:
        path = root / rel
        if path.exists():
            lines.append(f"{sha256_file(path)}  {rel}")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write the manifest and print it."""
    text = build_manifest()
    MANIFEST.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
