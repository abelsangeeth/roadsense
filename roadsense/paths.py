import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ROADSENSE_DATA", "E:/roadsense-env/data"))
CACHE = Path(os.environ.get("ROADSENSE_CACHE", "E:/roadsense-env/cache"))
RUNS = Path(os.environ.get("ROADSENSE_RUNS", ROOT / "runs"))
REPORTS = ROOT / "reports"
