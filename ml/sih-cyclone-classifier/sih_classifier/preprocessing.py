from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import tensorflow as tf


RAW_IMAGE_SIZE = 201
MODEL_IMAGE_SIZE = 170
INVALID_ABS_LIMIT = 1e30

MODALITIES = ("IR", "WV", "VIS")

DEFAULT_STATS_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "v4_normalization_stats.json"
)


def load_normalization_stats(
    path: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """
    Load the frozen V4 normalization statistics.

    Expected logical format:

        {
            "IR":  {"median": ..., "p01": ..., "p99": ...},
            "WV":  {"median": ..., "p01": ..., "p99": ...},
            "VIS": {"median": ..., "p01": ..., "p99": ...}
        }

    Also accepts the original V4 list-of-records format.
    """

    stats_path = (
        Path(path)
        if path is not None
        else DEFAULT_STATS_PATH
    )

    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Normalization statistics not found: {stats_path}"
        )

    raw = json.loads(
        stats_path.read_text(encoding="utf-8")
    )

    # Direct dictionary format
    if isinstance(raw, dict) and all(
        modality in raw for modality in MODALITIES
    ):
        records = raw

    # Original V4 list-of-records format
    elif isinstance(raw, list):
        records = {
            str(item["channel"]): item
            for item in raw
        }

        # V4 calls the infrared channel IR1.
        if "IR1" in records and "IR" not in records:
            records["IR"] = records["IR1"]

    else:
        raise ValueError(
            f"Unsupported normalization-statistics format: "
            f"{stats_path}"
        )

    required = {"median", "p01", "p99"}

    output = {}

    for modality in MODALITIES:

        key = modality

        if key not in records:
            if modality == "IR" and "IR1" in records:
                key = "IR1"
            else:
                raise ValueError(
                    f"Missing normalization statistics for {modality}"
                )

        record = records[key]

        if not required.issubset(record):
            raise ValueError(
                f"Incomplete normalization statistics for {modality}"
            )

        median = float(record["median"])
        p01 = float(record["p01"])
        p99 = float(record["p99"])

        if not (
            np.isfinite(median)
            and np.isfinite(p01)
            and np.isfinite(p99)
        ):
            raise ValueError(
                f"Non-finite normalization statistics for {modality}"
            )

        if p99 <= p01:
            raise ValueError(
                f"Invalid P01/P99 range for {modality}: "
                f"P01={p01}, P99={p99}"
            )

        output[modality] = {
            "median": median,
            "p01": p01,
            "p99": p99,
        }

    return output


def _validate_raw(
    raw: np.ndarray,
) -> np.ndarray:

    x = np.asarray(raw, dtype=np.float32)

    if x.ndim == 3:
        x = x[None, ...]

    if x.ndim != 4:
        raise ValueError(
            "Expected raw input shape "
            "(N,201,201,4) or (201,201,4); "
            f"got {x.shape}"
        )

    if x.shape[1:3] != (
        RAW_IMAGE_SIZE,
        RAW_IMAGE_SIZE,
    ):
        raise ValueError(
            "Expected raw spatial size 201x201; "
            f"got {x.shape[1:3]}"
        )

    if x.shape[-1] < 3:
        raise ValueError(
            "Raw input must contain at least "
            "IR, WV and VIS channels."
        )

    return x


def preprocess_raw(
    raw: np.ndarray,
    stats: Mapping[
        str,
        Mapping[str, float]
    ] | None = None,
) -> np.ndarray:
    """
    Convert raw TCIR-CPAC_IO_SH imagery to V7 input.

    Input:
        (201,201,4)
        or
        (N,201,201,4)

    Raw channel order:
        0 = IR
        1 = WV
        2 = VIS
        3 = PMW

    Output:
        (N,170,170,3)

    Processing:
        1. invalid/non-finite -> channel median
        2. clip to P01/P99
        3. map to [-1,1]
        4. resize 201x201 -> 170x170
    """

    x = _validate_raw(raw)

    norm = (
        stats
        if stats is not None
        else load_normalization_stats()
    )

    processed = np.empty(
        (
            x.shape[0],
            RAW_IMAGE_SIZE,
            RAW_IMAGE_SIZE,
            3,
        ),
        dtype=np.float32,
    )

    for i, modality in enumerate(MODALITIES):

        st = norm[modality]

        a = x[..., i]

        bad = (
            ~np.isfinite(a)
            | (np.abs(a) > INVALID_ABS_LIMIT)
        )

        a = np.where(
            bad,
            np.float32(st["median"]),
            a,
        )

        lo = np.float32(st["p01"])
        hi = np.float32(st["p99"])

        a = np.clip(a, lo, hi)

        processed[..., i] = (
            ((a - lo) / (hi - lo))
            * 2.0
            - 1.0
        )

    resized = tf.image.resize(
        processed,
        [
            MODEL_IMAGE_SIZE,
            MODEL_IMAGE_SIZE,
        ],
        method="bilinear",
        antialias=True,
    )

    return resized.numpy().astype(
        np.float32,
        copy=False,
    )


def preprocess_modalities(
    images: Mapping[str, np.ndarray],
    stats: Mapping[
        str,
        Mapping[str, float]
    ] | None = None,
) -> dict[str, np.ndarray]:
    """
    Preprocess separately supplied raw IR/WV/VIS imagery.
    """

    missing = [
        modality
        for modality in MODALITIES
        if modality not in images
    ]

    if missing:
        raise ValueError(
            f"Missing modalities: {missing}"
        )

    prepared = []

    for modality in MODALITIES:

        a = np.asarray(
            images[modality],
            dtype=np.float32,
        )

        if a.ndim == 2:
            a = a[None, ...]

        if a.ndim == 3:
            a = a[..., None]

        if (
            a.ndim != 4
            or a.shape[-1] != 1
            or a.shape[1:3]
            != (
                RAW_IMAGE_SIZE,
                RAW_IMAGE_SIZE,
            )
        ):
            raise ValueError(
                f"{modality}: expected "
                "(N,201,201) or "
                "(N,201,201,1), "
                f"got {a.shape}"
            )

        prepared.append(a[..., 0])

    stacked = np.stack(
        prepared,
        axis=-1,
    )

    x = preprocess_raw(
        stacked,
        stats=stats,
    )

    return {
        modality: x[..., i:i + 1]
        for i, modality in enumerate(MODALITIES)
    }


def validate_model_input(
    x: np.ndarray,
) -> np.ndarray:

    arr = np.asarray(
        x,
        dtype=np.float32,
    )

    if arr.ndim == 3:
        arr = arr[..., None]

    expected = (
        MODEL_IMAGE_SIZE,
        MODEL_IMAGE_SIZE,
        1,
    )

    if arr.ndim != 4 or arr.shape[1:] != expected:
        raise ValueError(
            f"Expected (N,170,170,1), "
            f"got {arr.shape}"
        )

    if not np.isfinite(arr).all():
        raise ValueError(
            "Model input contains NaN or Inf "
            "after preprocessing."
        )

    return arr