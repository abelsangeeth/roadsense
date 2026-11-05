"""Command-line interface.

    python -m roadsense.cli image frame.jpg [--mode scene|sign|surface]
    python -m roadsense.cli video drive.mp4 annotated.mp4 [--every 2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .advisory import advise
from .pipeline import RoadSense

GREEN, AMBER, WHITE = (122, 175, 27), (0, 161, 237), (255, 255, 255)  # BGR for OpenCV


def cmd_image(rs: RoadSense, a):
    res = rs.analyze(Image.open(a.input), mode=a.mode)
    print(json.dumps(res, indent=2))


def draw(frame, res, smoothed, adv):
    import cv2

    h, w = frame.shape[:2]
    s = max(1.0, w / 1000)
    if res["surface"] and res["surface"].get("roi"):
        x1, y1, x2, y2 = map(int, res["surface"]["roi"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, max(1, int(s)), cv2.LINE_AA)
    for sg in res["signs"]:
        x1, y1, x2, y2 = map(int, sg["box"])
        col = GREEN if sg["accepted"] else AMBER
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, int(3 * s), cv2.LINE_AA)
        txt = f"{sg['name'] if sg['accepted'] else 'uncertain'} {sg['conf']:.2f}"
        cv2.putText(frame, txt, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, col, max(1, int(s)), cv2.LINE_AA)
    # HUD
    top = max(smoothed, key=smoothed.get) if smoothed else None
    lines = [f"limit {adv['posted_limit_kmh'] or '-'}  advised {adv['recommended_kmh'] or '-'} km/h",
             f"surface {top.replace('_', ' ') if top else '-'}  mu {adv['mu'] if adv['mu'] is not None else '-'}",
             f"hazard {adv['hazard_label']}"]
    cv2.rectangle(frame, (0, 0), (int(330 * s), int(78 * s)), (20, 20, 20), -1)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (int(10 * s), int((22 + 22 * i) * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, WHITE, max(1, int(s)), cv2.LINE_AA)
    return frame


def cmd_video(rs: RoadSense, a):
    import cv2

    cap = cv2.VideoCapture(str(a.input))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    limit, smoothed, res, adv, n, log = None, None, None, None, 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % a.every == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = rs.analyze(img, mode="scene", current_limit=limit)
            if res["surface"]:
                f = res["surface"]["friction"]
                # EMA over frames: a single misread frame should not flip the advisory
                smoothed = f if smoothed is None else {k: 0.8 * smoothed[k] + 0.2 * f[k] for k in f}
                surf = {"friction": smoothed, "unevenness": res["surface"]["unevenness"]}
            else:
                surf = None
            adv = advise([s for s in res["signs"] if s["accepted"]], surf, limit)
            limit = adv["posted_limit_kmh"]
            log.append({"frame": n, "signs": [s["name"] for s in res["signs"] if s["accepted"]], **adv})
        out.write(draw(frame, res, smoothed, adv))
        n += 1
    cap.release()
    out.release()
    Path(a.output).with_suffix(".json").write_text(json.dumps(log, indent=1))
    print(f"wrote {a.output} ({n} frames) and {Path(a.output).with_suffix('.json')}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="roadsense")
    ap.add_argument("--release", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / "release")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("image")
    p.add_argument("input", type=Path)
    p.add_argument("--mode", choices=["scene", "sign", "surface"], default="scene")
    p = sub.add_parser("video")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--every", type=int, default=2, help="analyse every n-th frame (others reuse the overlay)")
    a = ap.parse_args(argv)
    rs = RoadSense.load(a.release)
    {"image": cmd_image, "video": cmd_video}[a.cmd](rs, a)


if __name__ == "__main__":
    main()
