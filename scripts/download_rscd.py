"""Download a stratified, seeded subsample of RSCD (rezzzq/RSCD-1million) from Hugging Face.

The repo stores ~1M loose JPGs and has no parquet export, and anonymous access is
rate-limited (tree listing: 500 req / 5 min, file resolves: 3000 req / 5 min). So we
list every class folder once (cached to manifest.json), sample uniformly at random
*across* the whole folder -- the files are timestamp-ordered video frames, so taking
the first N would give a handful of near-duplicate sequences -- and download with a
token bucket that stays under the resolver quota.

    python scripts/download_rscd.py --out E:/roadsense-env/data/rscd --train-per-class 1200
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "rezzzq/RSCD-1million"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def _get(url: str, retries: int = 8) -> tuple[bytes, dict]:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "roadsense/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", "30"))
                print(f"  429 rate limited, sleeping {wait}s", flush=True)
                time.sleep(wait + 1)
            elif e.code >= 500 or e.code in (408, 499):  # 499: CDN-side client-closed, transient
                time.sleep(2 ** attempt)
            else:
                raise
        except Exception:  # connection reset / timeout
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {url}")


def list_folder(path: str) -> list[str]:
    files: list[str] = []
    url = f"{API}/{urllib.parse.quote(path)}?limit=1000"
    while url:
        body, headers = _get(url)
        files += [e["path"] for e in json.loads(body) if e["type"] == "file"]
        link = headers.get("Link", "")
        url = link[1:link.index(">")] if 'rel="next"' in link else None
        time.sleep(0.62)  # 500 req / 300 s
    return files


def label_of(path: str) -> str:
    """'train/dry_asphalt_smooth/x.jpg' or 'test_50k/123-dry-asphalt-smooth.jpg' -> 'dry_asphalt_smooth'."""
    parts = path.split("/")
    if parts[0] == "train":
        return parts[1]
    stem = parts[-1].rsplit(".", 1)[0]
    return "_".join(stem.split("-")[1:])


def build_manifest(out: Path) -> dict[str, list[str]]:
    cache = out / "manifest.json"
    if cache.exists():
        return json.loads(cache.read_text())
    body, _ = _get(f"{API}/train")
    classes = [e["path"] for e in json.loads(body) if e["type"] == "directory"]
    manifest: dict[str, list[str]] = {}
    for c in classes:
        manifest[c] = list_folder(c)
        print(f"listed {c}: {len(manifest[c])}", flush=True)
    for split in ("vali_20k", "test_50k"):
        manifest[split] = list_folder(split)
        print(f"listed {split}: {len(manifest[split])}", flush=True)
    cache.write_text(json.dumps(manifest))
    return manifest


class Bucket:
    def __init__(self, rate: float):
        self.interval, self.next, self.lock = 1.0 / rate, time.monotonic(), threading.Lock()

    def take(self):
        with self.lock:
            now = time.monotonic()
            self.next = max(self.next, now) + self.interval
            delay = self.next - self.interval - now
        if delay > 0:
            time.sleep(delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-per-class", type=int, default=1200)
    ap.add_argument("--val-per-class", type=int, default=150)
    ap.add_argument("--test-per-class", type=int, default=400)
    ap.add_argument("--rate", type=float, default=9.5, help="files per second (quota is 10)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(a.out)
    rng = random.Random(a.seed)

    chosen: list[tuple[str, str]] = []  # (repo path, local path)
    for key, files in manifest.items():
        if key.startswith("train/"):
            groups = {label_of(key + "/x.jpg"): sorted(files)}
            split, k = "train", a.train_per_class
        else:
            groups = defaultdict(list)
            for f in sorted(files):
                groups[label_of(f)].append(f)
            split, k = ("val", a.val_per_class) if key == "vali_20k" else ("test", a.test_per_class)
        for label, fs in groups.items():
            for f in rng.sample(fs, min(k, len(fs))):
                chosen.append((f, str(a.out / split / label / Path(f).name)))

    todo = [(r, l) for r, l in chosen if not Path(l).exists()]
    print(f"selected {len(chosen)} files, {len(todo)} to download", flush=True)
    bucket, done, t0 = Bucket(a.rate), [0], time.time()

    def fetch(item):
        repo_path, local = item
        bucket.take()
        try:
            data, _ = _get(f"{RESOLVE}/{urllib.parse.quote(repo_path)}")
        except Exception as e:  # one bad file must not kill an hour-long download; rerun resumes
            print(f"  FAILED {repo_path}: {e}", flush=True)
            return
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        tmp = local + ".part"
        Path(tmp).write_bytes(data)
        Path(tmp).replace(local)
        done[0] += 1
        if done[0] % 1000 == 0:
            el = time.time() - t0
            print(f"  {done[0]}/{len(todo)}  {done[0]/el:.1f}/s  eta {(len(todo)-done[0])/(done[0]/el)/60:.0f} min", flush=True)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(fetch, todo))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
