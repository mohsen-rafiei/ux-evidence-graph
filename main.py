#!/usr/bin/env python3
"""Entry point for the ux-evidence-graph command-line interface.

Examples:
    python main.py init
    python main.py ingest --input data/sample
    python main.py extract
    python main.py export-review --output outputs/review/edges_for_review.csv
    python main.py ask --question "Which pain points are connected to trust?"
"""

from uxeg.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
