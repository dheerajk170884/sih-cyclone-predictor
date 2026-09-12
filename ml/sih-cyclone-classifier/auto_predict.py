#!/usr/bin/env python3
"""
Automatic SIH cyclone prediction runner.

Usage:

    python auto_predict.py demo_data\\demo_sample.npy

Or from another Python program:

    from auto_predict import predict_cyclone

    result = predict_cyclone("demo_data/demo_sample.npy")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sih_classifier import SIHClassifierPipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = PROJECT_ROOT / "models"
DEFAULT_STATS = PROJECT_ROOT / "configs" / "v4_normalization_stats.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "prediction.json"


def _convert_to_json(result: dict) -> dict:
    """Convert NumPy values into JSON-compatible Python objects."""

    output = {
        "final_vmax_kt": (
            np.asarray(result["final_vmax_kt"])
            .reshape(-1)
            .tolist()
        ),
        "final_class_index": (
            np.asarray(result["final_class_index"])
            .reshape(-1)
            .tolist()
        ),
        "final_class": list(result["final_class"]),
        "branches": {},
    }

    for modality, branch in result["branches"].items():
        output["branches"][modality] = {
            "probabilities": (
                np.asarray(branch["probabilities"])
                .tolist()
            ),
            "class_index": (
                np.asarray(branch["class_index"])
                .reshape(-1)
                .tolist()
            ),
            "vmax_kt": (
                np.asarray(branch["vmax_kt"])
                .reshape(-1)
                .tolist()
            ),
        }

    return output


def predict_cyclone(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    models_dir: str | Path = DEFAULT_MODELS,
    stats_path: str | Path = DEFAULT_STATS,
) -> dict:
    """
    Run the complete SIH classifier automatically.

    Parameters
    ----------
    input_path:
        Raw .npy satellite image.
        Expected shape:
            (201,201,4)
        or
            (N,201,201,4)

    output_path:
        JSON file where prediction results are saved.

    models_dir:
        Directory containing the 13 V7 .keras model files.

    stats_path:
        V4 normalization statistics JSON.

    Returns
    -------
    dict
        Prediction result.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input file not found:\n{input_path}"
        )

    # --------------------------------------------------
    # Load raw satellite image
    # --------------------------------------------------
    raw = np.load(
        input_path,
        allow_pickle=False,
    )

    print("=" * 70)
    print("SIH AUTOMATIC CYCLONE PREDICTION")
    print("=" * 70)
    print(f"Input : {input_path}")
    print(f"Shape : {raw.shape}")
    print()

    # --------------------------------------------------
    # Load classifier pipeline
    # --------------------------------------------------
    pipeline = SIHClassifierPipeline(
        models_dir=models_dir,
        normalization_stats=stats_path,
    )

    pipeline.load()

    # --------------------------------------------------
    # Run prediction
    # --------------------------------------------------
    result = pipeline.predict_raw(raw)

    json_result = _convert_to_json(result)

    # --------------------------------------------------
    # Save result
    # --------------------------------------------------
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            json_result,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Print final result
    # --------------------------------------------------
    print()
    print("=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)

    for i, (vmax, cyclone_class) in enumerate(
        zip(
            json_result["final_vmax_kt"],
            json_result["final_class"],
        )
    ):
        print(
            f"Sample {i}: "
            f"{vmax:.2f} kt -> {cyclone_class}"
        )

    print()
    print(f"Saved result: {output_path.resolve()}")
    print("=" * 70)

    return json_result


def main() -> None:
    import argparse

    DEFAULT_INPUT = (
        PROJECT_ROOT
        / "demo_data"
        / "demo_sample.npy"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Automatically run the SIH V7 cyclone "
            "intensity classifier."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=(
            "Raw satellite .npy file "
            "(default: ./demo_data/demo_sample.npy)"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=(
            "Output JSON file "
            "(default: prediction.json)"
        ),
    )

    args = parser.parse_args()

    predict_cyclone(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()