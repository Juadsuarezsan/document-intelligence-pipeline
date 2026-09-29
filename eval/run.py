"""Run the evaluation and regenerate ``eval/RESULTS.md``: ``python -m eval.run``."""

from __future__ import annotations

import sys

from src.eval.run_eval import main

if __name__ == "__main__":
    sys.exit(main())
