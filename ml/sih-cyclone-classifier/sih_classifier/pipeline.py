from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .model import V7Classifier
from .preprocessing import (
    load_normalization_stats,
    preprocess_modalities,
    preprocess_raw,
)


class SIHClassifierPipeline:
    """
    Portable end-to-end SIH V7 classifier pipeline.

    Flow:
        raw 201x201x4
            ↓
        V4 preprocessing
            ↓
        IR / WV / VIS 170x170x1
            ↓
        V7 TCIC
            ↓
        routed TCIE
            ↓
        branch Vmax predictions
            ↓
        fusion ANN
            ↓
        final Vmax + intensity class
    """

    def __init__(
        self,
        models_dir: str | Path | None = None,
        normalization_stats: str | Path | None = None,
    ):
        project_root = Path(__file__).resolve().parents[1]

        self.models_dir = (
            Path(models_dir).expanduser().resolve()
            if models_dir is not None
            else project_root / "models"
        )

        self.normalization_stats = (
            Path(normalization_stats).expanduser().resolve()
            if normalization_stats is not None
            else project_root
            / "configs"
            / "v4_normalization_stats.json"
        )

        self.classifier = V7Classifier(self.models_dir)
        self._loaded = False

    def load(self) -> None:
        """
        Validate preprocessing configuration and load all 13
        frozen V7 model files.
        """
        # Validate normalization statistics first.
        load_normalization_stats(self.normalization_stats)

        # Load all 13 Keras models.
        self.classifier.load()

        self._loaded = True

    def predict_raw(
        self,
        raw_images: np.ndarray,
    ) -> dict:
        """
        Predict directly from raw TCIR-CPAC_IO_SH imagery.

        Accepted input:
            (201,201,4)
            (N,201,201,4)

        Channel order:
            0 = IR
            1 = WV
            2 = VIS
            3 = PMW

        PMW is ignored by the V7 classifier.
        """
        if not self._loaded:
            self.load()

        stats = load_normalization_stats(
            self.normalization_stats
        )

        x = preprocess_raw(
            raw_images,
            stats=stats,
        )

        images = {
            "IR": x[..., 0:1],
            "WV": x[..., 1:2],
            "VIS": x[..., 2:3],
        }

        return self.classifier.predict(images)

    def predict_modalities(
        self,
        raw_images: Mapping[str, np.ndarray],
    ) -> dict:
        """
        Predict from separately supplied raw IR/WV/VIS images.

        Each modality may be:
            (201,201)
            (N,201,201)
            (201,201,1)
            (N,201,201,1)
        """
        if not self._loaded:
            self.load()

        stats = load_normalization_stats(
            self.normalization_stats
        )

        images = preprocess_modalities(
            raw_images,
            stats=stats,
        )

        return self.classifier.predict(images)