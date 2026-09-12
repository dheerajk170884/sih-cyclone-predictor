#!/usr/bin/env python3
"""Simple end-to-end SIH V7 classifier pipeline CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sih_classifier import SIHClassifierPipeline


def main():
    ap = argparse.ArgumentParser(
        description="SIH V7 raw-image classifier pipeline"
    )

    ap.add_argument(
        "--models",
        default=str(Path(__file__).resolve().parent / "models"),
        help=(
            "Directory containing the 13 .keras model files "
            "(default: ./models)"
        ),
    )

    ap.add_argument(
        "--stats",
        default=None,
        help=(
            "Path to V4 normalization statistics JSON "
            "(default: ./configs/v4_normalization_stats.json)"
        ),
    )

    ap.add_argument(
        "--input",
        required=True,
        help=".npy containing (N,201,201,4) raw images",
    )

    ap.add_argument(
        "--output",
        default="prediction.json",
        help="Output JSON path",
    )

    args = ap.parse_args()

    raw = np.load(args.input)

    pipe = SIHClassifierPipeline(
        models_dir=args.models,
        normalization_stats=args.stats,
    )

    result = pipe.predict_raw(raw)

    out = {
        "final_vmax_kt": result["final_vmax_kt"].tolist(),
        "final_class_index": result["final_class_index"].tolist(),
        "final_class": result["final_class"],
        "branches": {},
    }

    for modality, branch in result["branches"].items():
        out["branches"][modality] = {
            "probabilities": branch["probabilities"].tolist(),
            "class_index": branch["class_index"].tolist(),
            "vmax_kt": branch["vmax_kt"].tolist(),
        }

    Path(args.output).write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()