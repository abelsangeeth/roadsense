"""RoadSense web API + dashboard.

    python -m uvicorn serve.app:app --host 127.0.0.1 --port 8765
    # then open http://127.0.0.1:8765
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from roadsense.data.augment import normalize
from roadsense.eval.gradcam import signnet_cam
from roadsense.pipeline import RoadSense

ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path(os.environ.get("ROADSENSE_RELEASE", ROOT / "runs" / "release"))
WEB = ROOT / "frontend"

app = FastAPI(title="RoadSense", version="1.0")
rs = RoadSense.load(RELEASE)
MODEL_CARD = json.loads((RELEASE / "model_card.json").read_text()) if (RELEASE / "model_card.json").exists() else {}


def _png_b64(arr: np.ndarray) -> str:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _cam_overlay(crops: np.ndarray) -> list[str]:
    """Grad-CAM heatmap blended over each 48x48 sign crop, upscaled for display."""
    if len(crops) == 0:
        return []
    import matplotlib

    x = normalize(torch.from_numpy(crops).to(rs.dev).permute(0, 3, 1, 2).float().div(255))
    with torch.enable_grad():
        cam, _, _ = signnet_cam(rs.signnet, x)
    cmap = matplotlib.colormaps["inferno"]
    out = []
    for c, h in zip(crops, cam[:, 0].cpu().numpy()):
        heat = (cmap(h)[..., :3] * 255).astype(np.float32)
        blend = (0.55 * c.astype(np.float32) + 0.45 * heat).astype(np.uint8)
        out.append(_png_b64(np.asarray(Image.fromarray(blend).resize((96, 96), Image.NEAREST))))
    return out


@app.get("/api/health")
def health():
    return {"ok": True, "device": str(rs.dev), "detector": rs.detector is not None,
            "surface": rs.surface is not None, "damage": rs.damage is not None, "calibration": rs.calib}


@app.get("/api/model-card")
def model_card():
    return MODEL_CARD


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), mode: str = Form("scene"), current_limit: int | None = Form(None),
                  explain: bool = Form(True)):
    if mode not in ("scene", "sign", "surface"):
        raise HTTPException(400, "mode must be scene, sign or surface")
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not decode image")
    if max(img.size) > 2400:
        img.thumbnail((2400, 2400))
    res = rs.analyze(img, mode=mode, current_limit=current_limit)
    if res["signs"]:
        from roadsense.data.prepare import crop_sign
        crops = np.stack([crop_sign(img, s["box"]) for s in res["signs"]])
        cams = _cam_overlay(crops) if explain else [None] * len(crops)
        for s, c, cam in zip(res["signs"], crops, cams):
            s["crop"] = _png_b64(np.asarray(Image.fromarray(c).resize((96, 96), Image.BICUBIC)))
            s["cam"] = cam
    return JSONResponse(res)


@app.get("/api/samples")
def samples():
    d = WEB / "samples"
    if not d.exists():
        return []
    return [{"url": f"/samples/{p.name}", "mode": p.name.split("__")[0], "name": p.stem.split("__")[-1]}
            for p in sorted(d.glob("*.jpg"))]


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/samples", StaticFiles(directory=WEB / "samples", check_dir=False), name="samples")
app.mount("/reports", StaticFiles(directory=ROOT / "reports", check_dir=False), name="reports")
