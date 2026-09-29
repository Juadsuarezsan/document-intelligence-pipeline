"""Regenerate the evaluation set: ``python scripts/generate_synthetic.py``.

Writes ``data/eval/test_set.jsonl`` (synthetic documents + FUNSD sample, all with
ground truth) and ``data/eval/tables.jsonl``. Deterministic for a given seed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.dataset import TABLE_SET, TEST_SET, write_jsonl  # noqa: E402
from src.synth.documents import SEED, generate_documents, generate_tables  # noqa: E402
from src.synth.funsd import load_funsd_dir  # noqa: E402

FUNSD_DIR = Path(__file__).resolve().parents[1] / "data" / "funsd_sample" / "annotations"


def main() -> int:
    """Generate and write both JSONL files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--tables", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    docs = [d.to_record() for d in generate_documents(args.per_type, args.seed)]
    funsd = [d.to_record() for d in load_funsd_dir(FUNSD_DIR)] if FUNSD_DIR.exists() else []
    write_jsonl(TEST_SET, docs + funsd)
    write_jsonl(TABLE_SET, generate_tables(args.tables, args.seed + 1))
    print(f"wrote {len(docs)} synthetic + {len(funsd)} FUNSD cases to {TEST_SET}")
    print(f"wrote {args.tables} tables to {TABLE_SET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
