"""Evaluation of the road-surface models -> reports/surface/*.

    python -m roadsense.eval.evaluate_surface
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from ..data.augment import MEAN, STD, normalize, to_float
from ..labels import FRICTION, IGNORE, MATERIAL, RSCD_CLASSES, UNEVENNESS
from ..models.surfacenet import build_surface_model, joint_probs
from ..paths import REPORTS, RUNS
from ..train.common import batches, save_json
from ..train.train_surface import FACTORS, head_targets, load_split, score
from .calibration import ece, fit_temperature
from .corruptions import CORRUPTIONS
from .gradcam import surface_cam
from .plots import SERIES, confusion, plt

OUT = REPORTS / "surface"
VARIANTS = {"SE-ResNet from scratch, multi-head (128x192)": "cnn_multi",
            "EfficientNet-B0 flat 27-way (128x192)": "effnet_flat",
            "EfficientNet-B0 multi-head (128x192)": "effnet_multi",
            "EfficientNet-B0 multi-head (192x288)": "effnet_multi_192",
            "ConvNeXt-Tiny multi-head (192x288)": "convnext_multi_192"}
_DATA = {}


def day_data(hw):
    """(val_x, val_y, test_x, test_y) of the day-disjoint split at a model's input size.
    Every resolution cache holds the same files in the same order."""
    hw = tuple(hw)
    if hw not in _DATA:
        xva, yva, _ = load_split("val", "day", hw)
        xte, yte, _ = load_split("test", "day", hw)
        _DATA[hw] = (xva, yva, xte, yte)
    return _DATA[hw]


def load(run, dev):
    ck = torch.load(RUNS / "surface" / run / "best.pt", map_location=dev, weights_only=False)
    m = build_surface_model(ck["config"]["arch"], ck["config"]["head"], pretrained=False)
    m.load_state_dict(ck["model"])
    m.hw = tuple(ck["config"].get("hw", (128, 192)))
    return m.to(dev).eval().to(memory_format=torch.channels_last)


def predict(model, x_u8, bs=256):
    return predict_float(model, to_float(x_u8), bs)


@torch.no_grad()
def predict_float(model, xf_cpu, bs=256):
    dev = next(model.parameters()).device
    outs = {}
    for idx in batches(len(xf_cpu), bs, False):
        x = normalize(xf_cpu[idx].to(dev)).contiguous(memory_format=torch.channels_last)
        o = {k: v.float() for k, v in model(x).items()}  # fp32, matching serving
        o["joint_p"] = joint_probs(o)
        for k, v in o.items():
            outs.setdefault(k, []).append(v)
    return {k: torch.cat(v) for k, v in outs.items()}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, _, yte = day_data((128, 192))
    report = {"protocol": "capture-day-disjoint", "variants": {}, "n_test": len(yte), "test_class_counts": np.bincount(yte.numpy(), minlength=27).tolist()}
    models = {}
    for name, run in VARIANTS.items():
        if (RUNS / "surface" / run / "best.pt").exists():
            models[name] = load(run, dev)
            report["variants"][name] = score(predict(models[name], day_data(models[name].hw)[2]), yte)
            print(name, report["variants"][name])
    off = RUNS / "surface" / "effnet_multi_official" / "metrics.json"
    if off.exists():  # same recipe, RSCD's own (frame-level, leaky) split: quantifies the leakage
        report["official_split_effnet_multi"] = {k[5:]: v for k, v in json.loads(off.read_text()).items() if k.startswith("test_")}
    # release model = best *validation* macro-F1 (recorded at training time); test is report-only
    val_f1 = {k: json.loads((RUNS / "surface" / VARIANTS[k] / "metrics.json").read_text())["val_best_joint_macro_f1"]
              for k in report["variants"]}
    report["val_joint_macro_f1"] = val_f1
    best_name = max(val_f1, key=val_f1.get)
    report["best"] = best_name
    m = models[best_name]
    xva, yva, xte, _ = day_data(m.hw)
    out = predict(m, xte)
    pj = out["joint_p"].argmax(1).cpu().numpy()
    y = yte.numpy()

    # ------------------------------------------------ confusion matrices
    fig, axs = plt.subplots(1, 4, figsize=(19, 5.2), gridspec_kw={"width_ratios": [2.4, 1, 1, 1]})
    confusion(axs[0], confusion_matrix(y, pj, labels=range(27)), [c.replace("_", " ") for c in RSCD_CLASSES], "27 classes")
    pf = FACTORS[pj]
    t = FACTORS[y]
    for k, (h, labels) in enumerate((("friction", FRICTION), ("material", MATERIAL), ("unevenness", UNEVENNESS))):
        keep = t[:, k] != IGNORE
        cm = confusion_matrix(t[keep, k], np.where(pf[keep, k] == IGNORE, 0, pf[keep, k]), labels=range(len(labels)))
        confusion(axs[k + 1], cm, labels, h, annotate_off_diag=True)
        for i in range(len(labels)):
            axs[k + 1].text(i, i, f"{cm[i, i] / max(cm[i].sum(), 1) * 100:.0f}%", ha="center", va="center", fontsize=7, color="#fcfcfb")
    fig.suptitle(f"{best_name} on RSCD test (n={len(y)})", fontsize=10)
    fig.savefig(OUT / "confusion.png")
    plt.close(fig)
    per_class_f1 = f1_score(y, pj, average=None, labels=range(27))
    report["per_class_f1"] = dict(zip(RSCD_CLASSES, map(float, per_class_f1)))

    # ------------------------------------------------ calibration (27-way, on log of joint probs)
    lv = torch.log(predict(m, xva)["joint_p"].clamp_min(1e-9))
    T = fit_temperature(lv, yva.to(dev))
    lt = torch.log(out["joint_p"].clamp_min(1e-9))
    report["calibration"] = {"temperature": T,
                             "ece_raw": ece(lt.softmax(1).cpu().numpy(), y),
                             "ece_calibrated": ece((lt / T).softmax(1).cpu().numpy(), y)}
    print(report["calibration"])

    # ------------------------------------------------ robustness
    rob = {}
    # Corrupt once at the largest resolution, then resample to each model's input size:
    # kernels are in pixels, so corrupting each resolution separately would hit the
    # high-res models with a physically weaker blur / smaller occluder.
    hi = max({mm.hw for mm in models.values()})
    xf = to_float(day_data(hi)[2])
    for c in CORRUPTIONS:
        accs = {n: [] for n in models}
        for s in range(3):
            xc = torch.cat([CORRUPTIONS[c](xf[idx].to(dev), s, seed=17 * s + i).cpu() for i, idx in enumerate(batches(len(xf), 512, False))])
            for n, mm in models.items():
                xm = xc if mm.hw == hi else torch.cat([F.interpolate(xc[idx], size=mm.hw, mode="bilinear", antialias=True, align_corners=False)
                                                      for idx in batches(len(xc), 1024, False)])
                accs[n].append(score(predict_float(mm, xm), yte)["friction_acc"])
        rob[c] = accs
        print("robustness", c, {n: [round(a, 3) for a in v] for n, v in accs.items()})
    report["robustness_friction_acc"] = rob

    fig, ax = plt.subplots(figsize=(10, 3.6))
    names = list(CORRUPTIONS)
    w = 0.8 / len(models)
    for k, n in enumerate(models):
        ax.bar(np.arange(len(names)) + k * w, [np.mean(rob[c][n]) * 100 for c in names], w * 0.92, label=n, color=SERIES[k])
    ax.set_xticks(np.arange(len(names)) + w * (len(models) - 1) / 2, [c.replace("_", " ") for c in names])
    ax.set(ylabel="friction-state accuracy (%)", title="Surface friction-state accuracy under corruption (mean of 3 severities)", ylim=(0, 100))
    ax.axhline(report["variants"][best_name]["friction_acc"] * 100, color="#9a9893", lw=1, ls=":")
    ax.legend(loc="lower left")
    fig.savefig(OUT / "robustness.png")
    plt.close(fig)

    # ------------------------------------------------ Grad-CAM gallery (one per friction state)
    m32 = load(VARIANTS[best_name], dev).float().to(memory_format=torch.contiguous_format)
    pick = []
    for f in range(len(FRICTION)):
        cands = np.where((FACTORS[y][:, 0] == f) & (FACTORS[pj][:, 0] == f))[0]
        if len(cands):
            pick.append(int(np.random.default_rng(f).choice(cands)))
    x = normalize(to_float(xte[pick]).to(dev))
    cam, _ = surface_cam(m32, x, "friction" if "friction" in m32(x[:1]) else "joint")
    img = (x * STD.to(dev) + MEAN.to(dev)).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    fig, axs = plt.subplots(2, len(pick), figsize=(len(pick) * 2.2, 3.4))
    for k, i in enumerate(pick):
        axs[0, k].imshow(img[k])
        axs[0, k].set_title(RSCD_CLASSES[y[i]].replace("_", " "), fontsize=7)
        axs[1, k].imshow(img[k])
        axs[1, k].imshow(cam[k, 0].detach().cpu().numpy(), cmap="inferno", alpha=0.5)
        axs[0, k].axis("off")
        axs[1, k].axis("off")
    fig.suptitle("Grad-CAM for the friction head (correct predictions)", fontsize=9)
    fig.savefig(OUT / "explain_gallery.png")
    plt.close(fig)

    save_json(report, OUT / "report.json")
    save_json({"surface_temperature": 1.0, "surface_joint_temperature": T, "surface_run": VARIANTS[best_name]}, OUT / "calibration.json")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
