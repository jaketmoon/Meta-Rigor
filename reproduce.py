"""Run the frozen-input experiments from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend/src"))

from metarigor.experiments.reproduce import ROOT, json_paths, main, read

if __name__ == "__main__":
    main()
