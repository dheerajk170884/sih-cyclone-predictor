#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from sih_classifier import V7Classifier, load_normalization_stats, normalize_resize


def main():
    ap = argparse.ArgumentParser(description="SIH V7 cyclone intensity classifier")
    ap.add_argument("--models", required=True, help="Root containing IR/WV/VIS + fusion checkpoints")
    ap.add_argument("--normalization-stats", required=True, help="v4_normalization_stats.json")
    ap.add_argument("--input", required=True, help="Raw .npy array: (N,201,201,4) or preprocessed (N,170,170,3)")
    ap.add_argument("--output", default="prediction.json")
    args = ap.parse_args()

    raw = np.load(args.input)
    if raw.ndim != 4:
        raise ValueError(f"Input must be 4-D, got {raw.shape}")
    if raw.shape[1:3] == (201, 201):
        x = normalize_resize(raw, load_normalization_stats(args.normalization_stats))
    elif raw.shape[1:3] == (170, 170) and raw.shape[-1] >= 3:
        x = raw[..., :3].astype(np.float32)
    else:
        raise ValueError(f"Unsupported input shape: {raw.shape}")

    images = {"IR": x[..., 0:1], "WV": x[..., 1:2], "VIS": x[..., 2:3]}
    model = V7Classifier(args.models)
    model.load()
    result = model.predict(images)

    serializable = {
        "final_vmax_kt": result["final_vmax_kt"].tolist(),
        "final_class_index": result["final_class_index"].tolist(),
        "final_class": result["final_class"],
        "branches": {},
    }
    for modality, branch in result["branches"].items():
        serializable["branches"][modality] = {
            "probabilities": branch["probabilities"].tolist(),
            "class_index": branch["class_index"].tolist(),
            "vmax_kt": branch["vmax_kt"].tolist(),
        }
    Path(args.output).write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
