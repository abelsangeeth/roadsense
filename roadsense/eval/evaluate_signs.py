"""Full evaluation of the traffic-sign models -> reports/signs/*.json + figures.

    python -m roadsense.eval.evaluate_signs
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from ..data.augment import MEAN, STD, normalize, to_float
from ..labels import GTSRB_CLASSES, SIGN_GROUP
from ..models.signnet import SignNet
from ..paths import REPORTS, RUNS
from ..train.common import batches, save_json
from ..train.train_signs import load_gtsrb
from .calibration import ece, fit_temperature, risk_coverage, threshold_for_risk
from .corruptions import CORRUPTIONS
from .gradcam import signnet_cam
from .plots import SERIES, plt, confusion

OUT = REPORTS / "signs"
VARIANTS = {  # name -> run dirs (seeds)
    "SignNet (full)": ["full_s0", "full_s1", "full_s2"],
    "- STN": ["ablate_nostn"],
    "- SE attention": ["ablate_nose"],
    "basic aug only": ["ablate_augbasic"],
    "no augmentation": ["ablate_augnone"],
}


def load_model(run: Path, dev):
    ck = torch.load(run / "best.pt", map_location=dev, weights_only=False)
    c = ck["config"]
    m = SignNet(width=c["width"], use_stn=not c["no_stn"], use_se=not c["no_se"])
    m.load_state_dict(ck["model"])
    return m.to(dev).eval().to(memory_format=torch.channels_last)


@torch.no_grad()
def logits_of(model, x_float, bs=1024):
    out = []
    for idx in batches(len(x_float), bs, False):
        # fp32 on purpose: serving runs fp32, and bf16 flips 0-3 of the 12,630 test predictions
        x = normalize(x_float[idx]).contiguous(memory_format=torch.channels_last)
        out.append(model(x).float())
    return torch.cat(out)


def corrupted(x_u8, name, sev, bs=2048):
    parts = []
    for i, idx in enumerate(batches(len(x_u8), bs, False)):
        parts.append(CORRUPTIONS[name](to_float(x_u8[idx.to(x_u8.device)]), sev, seed=1000 * sev + i))
    return torch.cat(parts)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_gtsrb(dev)
    xv, yv = data["val"]
    xt, yt = data["test"]
    ytn = yt.cpu().numpy()
    report = {"variants": {}}

    # ---------------------------------------------------- clean accuracy, all variants
    models = {}
    for name, runs in VARIANTS.items():
        accs, f1s = [], []
        for r in runs:
            p = RUNS / "signs" / r
            if not (p / "best.pt").exists():
                continue
            m = load_model(p, dev)
            pred = logits_of(m, to_float(xt)).argmax(1).cpu().numpy()
            accs.append((pred == ytn).mean())
            f1s.append(f1_score(ytn, pred, average="macro"))
            models.setdefault(name, m)
        if accs:
            report["variants"][name] = {"test_acc_mean": float(np.mean(accs)), "test_acc_std": float(np.std(accs)),
                                        "test_macro_f1_mean": float(np.mean(f1s)), "seeds": len(accs),
                                        "test_acc_runs": [float(a) for a in accs]}
            print(name, report["variants"][name])

    main_model = models["SignNet (full)"]
    lt, lv = logits_of(main_model, to_float(xt)), logits_of(main_model, to_float(xv))
    pred = lt.argmax(1).cpu().numpy()

    # ---------------------------------------------------- per-class / group
    cm = confusion_matrix(ytn, pred, labels=range(43))
    per_class = {GTSRB_CLASSES[i]: float(cm[i, i] / cm[i].sum()) for i in range(43)}
    groups = {}
    for g in sorted(set(SIGN_GROUP.values())):
        ids = [i for i in range(43) if SIGN_GROUP[i] == g]
        msk = np.isin(ytn, ids)
        groups[g] = {"acc": float((pred[msk] == ytn[msk]).mean()), "n": int(msk.sum())}
    errs = [(GTSRB_CLASSES[i], GTSRB_CLASSES[j], int(cm[i, j])) for i in range(43) for j in range(43) if i != j and cm[i, j]]
    report["main"] = {"test_acc": float((pred == ytn).mean()), "errors": int((pred != ytn).sum()),
                      "per_group": groups, "worst_classes": sorted(per_class.items(), key=lambda kv: kv[1])[:6],
                      "top_confusions": sorted(errs, key=lambda e: -e[2])[:10]}

    fig, ax = plt.subplots(figsize=(9, 8.5))
    confusion(ax, cm, [f"{i}" for i in range(43)], f"GTSRB test confusion (acc {report['main']['test_acc']*100:.2f}%, red = error counts)")
    fig.savefig(OUT / "confusion.png")
    plt.close(fig)

    # ---------------------------------------------------- calibration + selective prediction
    T = fit_temperature(lv, yv)
    pv, pt_raw, pt = (lv / T).softmax(1).cpu().numpy(), lt.softmax(1).cpu().numpy(), (lt / T).softmax(1).cpu().numpy()
    yvn = yv.cpu().numpy()
    thr = threshold_for_risk(pv, yvn, 0.001)
    acc_mask = pt.max(1) >= thr
    report["calibration"] = {
        "temperature": T, "ece_raw": ece(pt_raw, ytn), "ece_calibrated": ece(pt, ytn),
        "accept_threshold_val_risk_0.1pct": thr,
        "test_coverage_at_threshold": float(acc_mask.mean()),
        "test_error_on_accepted": float((pt.argmax(1)[acc_mask] != ytn[acc_mask]).mean()),
        "test_errors_rejected": int(((pt.argmax(1) != ytn) & ~acc_mask).sum()),
    }
    print(report["calibration"])
    cov, risk, _ = risk_coverage(pt, ytn)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].plot(cov * 100, risk * 100, color=SERIES[0])
    axs[0].axvline(acc_mask.mean() * 100, color=SERIES[1], lw=1, ls="--", label=f"operating point (thr {thr:.2f})")
    axs[0].set(xlabel="coverage (% of signs accepted)", ylabel="error among accepted (%)", title="Risk-coverage (test)", xlim=(80, 100.5))
    axs[0].legend()
    bins = np.linspace(0, 1, 16)
    for p, lab, col in ((pt_raw, "raw", SERIES[1]), (pt, f"T={T:.2f}", SERIES[0])):
        conf, ok = p.max(1), p.argmax(1) == ytn
        idx = np.digitize(conf, bins) - 1
        xs = [conf[idx == b].mean() for b in range(15) if (idx == b).sum() > 20]
        ys = [ok[idx == b].mean() for b in range(15) if (idx == b).sum() > 20]
        axs[1].plot(xs, ys, marker="o", ms=4, color=col, label=f"{lab} (ECE {ece(p, ytn)*100:.2f}%)")
    axs[1].plot([0, 1], [0, 1], color="#9a9893", lw=1, ls=":")
    axs[1].set(xlabel="confidence", ylabel="accuracy", title="Reliability diagram (test)")
    axs[1].legend()
    fig.savefig(OUT / "calibration.png")
    plt.close(fig)

    # ---------------------------------------------------- robustness
    rob = {}
    for cname in CORRUPTIONS:
        for sev in range(3):
            xc = corrupted(xt, cname, sev)
            for vname, m in models.items():
                acc = float((logits_of(m, xc).argmax(1) == yt).float().mean())
                rob.setdefault(vname, {}).setdefault(cname, []).append(acc)
        print("robustness", cname, {k: [round(a, 4) for a in v[cname]] for k, v in rob.items()})
    report["robustness"] = {v: {"per_corruption": d, "mean_corrupted_acc": float(np.mean([a for s in d.values() for a in s]))}
                            for v, d in rob.items()}
    names = list(CORRUPTIONS)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    shown = [v for v in ("SignNet (full)", "basic aug only", "no augmentation") if v in rob]
    w = 0.8 / len(shown)
    for k, v in enumerate(shown):
        ax.bar(np.arange(len(names)) + k * w, [np.mean(rob[v][c]) * 100 for c in names], w * 0.92, label=v, color=SERIES[k])
    ax.set_xticks(np.arange(len(names)) + w * (len(shown) - 1) / 2, [n.replace("_", " ") for n in names])
    ax.set(ylabel="accuracy (%), mean of 3 severities", title="Robustness of sign recognition to road-scene corruptions (GTSRB test)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left")
    fig.savefig(OUT / "robustness.png")
    plt.close(fig)

    # ---------------------------------------------------- explainability gallery
    rng = np.random.default_rng(0)
    wrong = np.where(pred != ytn)[0]
    pick = np.concatenate([rng.choice(np.where(pred == ytn)[0], 8, replace=False), wrong[:8]])
    x = normalize(to_float(xt[torch.from_numpy(pick).to(dev)])).float()
    m32 = load_model(RUNS / "signs" / VARIANTS["SignNet (full)"][0], dev).float().to(memory_format=torch.contiguous_format)
    cam, logits, theta = signnet_cam(m32, x)
    with torch.no_grad():
        warped, _ = m32.stn(x)
    unnorm = lambda t: (t * STD.to(dev) + MEAN.to(dev)).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    img, wimg, cam = unnorm(x), unnorm(warped), cam[:, 0].cpu().numpy()
    fig, axs = plt.subplots(3, len(pick), figsize=(len(pick) * 1.1, 3.9))
    for k in range(len(pick)):
        p, t = int(logits[k].argmax()), int(ytn[pick[k]])
        axs[0, k].imshow(img[k])
        axs[0, k].set_title(f"{'OK' if p == t else 'ERR'} {p}/{t}", fontsize=6, color=TEXT_OK if p == t else "#e34948")
        axs[1, k].imshow(wimg[k])
        axs[2, k].imshow(img[k])
        axs[2, k].imshow(cam[k], cmap="inferno", alpha=0.5)
        for r in range(3):
            axs[r, k].axis("off")
    axs[0, 0].text(-8, 24, "input", rotation=90, fontsize=7, va="center")
    axs[1, 0].text(-8, 24, "STN out", rotation=90, fontsize=7, va="center")
    axs[2, 0].text(-8, 24, "Grad-CAM", rotation=90, fontsize=7, va="center")
    fig.suptitle("Correct (left 8) and misclassified (right 8) test signs: predicted/true id", fontsize=9)
    fig.savefig(OUT / "explain_gallery.png")
    plt.close(fig)

    save_json(report, OUT / "report.json")
    save_json({"sign_temperature": T, "sign_accept_threshold": thr}, OUT / "calibration.json")
    print("wrote", OUT)


TEXT_OK = "#0b0b0b"

if __name__ == "__main__":
    main()
