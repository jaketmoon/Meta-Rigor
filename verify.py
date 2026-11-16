"""Verify the frozen inputs from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend/src"))

from metarigor.experiments.verify import main

if __name__ == "__main__":
    main()
