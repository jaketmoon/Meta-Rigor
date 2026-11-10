"""Run the CLI baselines from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend/src"))

from metarigor.experiments.baseline import ec_full_packages, main, run

if __name__ == "__main__":
    raise SystemExit(main())
