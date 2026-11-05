"""Collect every evaluation JSON into runs/release/model_card.json (rendered by the dashboard).

    python scripts/build_model_card.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roadsense.paths import REPORTS, RUNS  # noqa: E402

J = lambda p: json.loads(Path(p).read_text())
pct = lambda v, d=2: f"{v * 100:.{d}f}%"


def main():
    signs, surf = J(REPORTS / "signs" / "report.json"), J(REPORTS / "surface" / "report.json")
    e2e = J(REPORTS / "e2e" / "report_test.json")
    det = J(RUNS / "detector" / "metrics.json")
    dmg = J(REPORTS / "damage" / "metrics.json")
    dt = dmg["test"]
    full = signs["variants"]["SignNet (full)"]
    best = surf["variants"][surf["best"]]
    card = {
        "headline": [
            {"value": pct(full["test_acc_mean"]), "label": f"GTSRB test accuracy, 12,630 signs (mean of {full['seeds']} seeds)"},
            {"value": pct(e2e["detection_class_agnostic"]["AP50"], 1), "label": "Sign detection AP@0.5, GTSDB test scenes"},
            {"value": pct(e2e["operating_point"]["recognition_rate"], 1), "label": "End-to-end recognition rate (found + named), GTSDB test"},
            {"value": pct(best["friction_acc"], 1), "label": "Road friction-state accuracy, unseen capture days"},
            {"value": f"{dt['at_val_threshold']['F1']:.2f}", "label": "Dashcam road-damage F1 (cracks + potholes), RDD2022 held-out frames"},
            {"value": f"{e2e['fps']:.0f} fps", "label": f"Sign + surface pipeline throughput, before the damage detector was added ({'RTX 3060 laptop' if 'cuda' in e2e['device'] else 'CPU'})"},
        ],
        "tables": [
            {"title": "Traffic-sign recognition: ablation (GTSRB official test)",
             "columns": ["variant", "accuracy", "macro-F1", "accuracy under corruption"],
             "rows": [[k, pct(v["test_acc_mean"]) + (f" ± {v['test_acc_std']*100:.2f}" if v["seeds"] > 1 else ""),
                       f"{v['test_macro_f1_mean']:.4f}", pct(signs["robustness"].get(k, {}).get("mean_corrupted_acc", float("nan")), 1)]
                      for k, v in signs["variants"].items()],
             "note": "Corruption column: mean over 9 road-scene corruptions x 3 severities (noise, motion blur, defocus, fog, rain, low light, glare, occlusion, JPEG)."},
            {"title": "Road-surface condition (RSCD, capture-day-disjoint test)",
             "columns": ["model", "27-class acc", "27-class macro-F1", "friction", "material", "unevenness"],
             "rows": [[k, pct(v["joint_acc"], 1), f"{v['joint_macro_f1']:.3f}", pct(v["friction_acc"], 1), pct(v["material_acc"], 1), pct(v["unevenness_acc"], 1)]
                      for k, v in surf["variants"].items()],
             "note": ("Official RSCD split, same EfficientNet multi-head recipe: "
                      + (f"27-class acc {pct(surf['official_split_effnet_multi']['joint_acc'], 1)}, friction {pct(surf['official_split_effnet_multi']['friction_acc'], 1)}. "
                         if "official_split_effnet_multi" in surf else "")
                      + "84% of official test images share their capture second with a training image, so the day-disjoint numbers are the honest estimate.")},
            {"title": "Scene pipeline (GTSDB test, never used to train SignNet)",
             "columns": ["metric", "value"],
             "rows": [["detection AP@0.5 (class-agnostic)", pct(e2e["detection_class_agnostic"]["AP50"], 1)],
                      ["detection AP@[.5:.95]", pct(e2e["detection_class_agnostic"]["AP50:95"], 1)],
                      ["43-class AP@0.5 (detector x SignNet)", pct(e2e["detection_class_aware_43"]["AP50"], 1)],
                      ["recall at operating threshold", pct(e2e["operating_point"]["recall"], 1)],
                      ["classification accuracy of found signs", pct(e2e["operating_point"]["classification_acc_of_found"], 1)],
                      ["false positives / image", f"{e2e['operating_point']['false_positives_per_image']:.2f}"],
                      ["median latency", f"{e2e['latency_ms_median']:.0f} ms"],
                      ["detector val AP@0.5 / AP@[.5:.95]", f"{pct(det['val']['AP50'], 1)} / {pct(det['val']['AP50:95'], 1)}"]],
             "note": f"Test split: {e2e['n_images']} scenes, {e2e['n_gt']} signs (small; val numbers shown for stability)."},
            {"title": "Road damage from the dashcam (RDD2022, held-out frames from 6 countries)",
             "columns": ["class", "AP@0.5", "boxes"],
             "rows": [[k, f"{v['AP50']:.3f}", v["n_gt"]] for k, v in dt["per_class"].items()]
                     + [["all classes: mAP@0.5 / F1", f"{dt['mAP50']:.3f} / {dt['at_val_threshold']['F1']:.3f}", sum(v["n_gt"] for v in dt["per_class"].values())]],
             "note": (f"F1 at the validation-chosen threshold {dmg['threshold']:.2f}: precision {pct(dt['at_val_threshold']['precision'], 1)}, recall {pct(dt['at_val_threshold']['recall'], 1)}. By country: "
                      + ", ".join(f"{k.replace('_', ' ')} {v['F1_at_val_threshold']:.2f}" for k, v in dmg["test_by_country"].items())
                      + f". On {dmg['gtsdb_german_frames']['gtsdb_test_frames']} unseen German frames it flags damage on {dmg['gtsdb_german_frames']['frames_with_any_damage']} and reports no potholes.")},
            {"title": "Calibration and abstention (SignNet)",
             "columns": ["metric", "value"],
             "rows": [["temperature (fitted on val)", f"{signs['calibration']['temperature']:.3f}"],
                      ["ECE raw -> calibrated", f"{pct(signs['calibration']['ece_raw'])} -> {pct(signs['calibration']['ece_calibrated'])}"],
                      ["accept threshold (val risk <= 0.1%)", f"{signs['calibration']['accept_threshold_val_risk_0.1pct']:.3f}"],
                      ["test coverage at threshold", pct(signs["calibration"]["test_coverage_at_threshold"])],
                      ["test error on accepted signs", pct(signs["calibration"]["test_error_on_accepted"], 3)],
                      ["test errors caught by abstention", f"{signs['calibration']['test_errors_rejected']} of {signs['main']['errors']}"]]},
        ],
        "figures": [
            {"url": "/reports/signs/confusion.png", "caption": "SignNet confusion matrix on GTSRB test (off-diagonal counts in red)."},
            {"url": "/reports/signs/robustness.png", "caption": "Sign accuracy under road-scene corruptions: the augmentation pipeline is what buys robustness."},
            {"url": "/reports/signs/calibration.png", "caption": "Risk-coverage and reliability: abstaining on the least confident signs."},
            {"url": "/reports/signs/explain_gallery.png", "caption": "Input, spatial-transformer output and Grad-CAM for correct and misclassified signs."},
            {"url": "/reports/surface/confusion.png", "caption": "Road-surface confusion matrices on unseen capture days."},
            {"url": "/reports/surface/robustness.png", "caption": "Friction-state accuracy under corruption."},
            {"url": "/reports/surface/explain_gallery.png", "caption": "Grad-CAM of the friction head."},
            {"url": "/reports/damage/examples.jpg", "caption": "Damage detector: red = prediction, blue = ground truth. Top two rows: RDD2022 test frames; bottom row: German GTSDB frames with no damage labels."},
        ] + [{"url": f"/reports/e2e/{p.name}", "caption": f"GTSDB test scene {p.stem[6:]}: green = detected and correctly named, red = error, white = ground truth."}
             for p in sorted((REPORTS / "e2e").glob("scene_*.jpg"))[:3]],
    }
    (RUNS / "release" / "model_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card["headline"], indent=1))


if __name__ == "__main__":
    main()
