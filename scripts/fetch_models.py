"""Download the trained model weights into runs/release (they are too large for git).

    python scripts/fetch_models.py
"""
from __future__ import annotations

import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

REPO = "abelsangeeth/roadsense"
FILES = ["signnet.pt", "detector.pt", "surface.pt", "damage.pt"]
DEST = Path(__file__).resolve().parents[1] / "runs" / "release"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        out = DEST / f
        if out.exists():
            print(f"{f}: already present")
            continue
        url = f"https://github.com/{REPO}/releases/latest/download/{f}"
        print(f"{f}: downloading {url}")
        tmp = out.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, tmp)
        except urllib.error.HTTPError as e:
            # a private repository's release assets need an authenticated download
            if e.code != 404 or not shutil.which("gh"):
                raise SystemExit(f"{f}: download failed ({e}). If the repository is private, install the GitHub CLI, "
                                 f"run `gh auth login`, then rerun this script.")
            subprocess.run(["gh", "release", "download", "--repo", REPO, "--pattern", f, "--dir", str(DEST)], check=True)
            continue
        tmp.rename(out)
        print(f"{f}: {out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
