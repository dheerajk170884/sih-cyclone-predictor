from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow import keras

from .architecture import build_paper_tcic, build_paper_tcie
from .preprocessing import validate_model_input


CLASS_NAMES = (
    "TS_STS",
    "STY",
    "VSTY_ViolentTY",
)

MODALITIES = (
    "IR",
    "WV",
    "VIS",
)


def build_fusion():
    """
    Exact V7 fusion ANN architecture.

    Input:
        [IR_Vmax, WV_Vmax, VIS_Vmax]

    Architecture:
        Dense(8, relu)
        Dense(4, relu)
        Dense(1)
    """
    inp = keras.Input(
        shape=(3,),
        name="IR_WV_VIS_predictions",
    )

    x = keras.layers.Dense(
        8,
        activation="relu",
        name="fusion_dense8",
    )(inp)

    x = keras.layers.Dense(
        4,
        activation="relu",
        name="fusion_dense4",
    )(x)

    out = keras.layers.Dense(
        1,
        name="final_vmax",
    )(x)

    return keras.Model(
        inp,
        out,
        name="ThreeBranchFusionANN",
    )


def _load_weights_into_model(
    model: keras.Model,
    path: Path,
) -> keras.Model:
    """
    Load trained weights from the supplied .keras file.

    We intentionally rebuild the architecture instead of calling
    keras.models.load_model(), because the original TCIC architecture
    contains Lambda layers whose serialized output shape cannot be
    inferred by the current Keras loader.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    try:
        model.load_weights(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Could not load weights from:\n"
            f"  {path}\n\n"
            f"Rebuilt model:\n"
            f"  {model.name}\n"
            f"  parameters: {model.count_params():,}\n\n"
            f"Original error:\n"
            f"{exc}"
        ) from exc

    return model


class V7Classifier:
    """
    Frozen V7 IR/WV/VIS cyclone-intensity predictor.

    Each modality uses:

        TCIC
          ↓
        predicted class
          ↓
        routed TCIE expert
          ↓
        branch Vmax

    IR/WV/VIS branch Vmax values then enter:

        Fusion ANN
          ↓
        final Vmax
          ↓
        final intensity class
    """

    def __init__(
        self,
        model_root: str | Path,
    ):
        self.root = (
            Path(model_root)
            .expanduser()
            .resolve()
        )

        self.tcic: dict[str, keras.Model] = {}
        self.tcie: dict[
            tuple[str, int],
            keras.Model,
        ] = {}

        self.fusion: keras.Model | None = None

        self._loaded = False

    def _required_model_paths(self) -> list[Path]:
        required = []

        for modality in MODALITIES:
            required.append(
                self.root
                / f"{modality}_TCIC.keras"
            )

            for cls in range(3):
                required.append(
                    self.root
                    / f"{modality}_TCIE_class{cls}.keras"
                )

        required.append(
            self.root
            / "fusion_ann_best.keras"
        )

        return required

    def _validate_model_files(self) -> None:
        missing = [
            path
            for path in self._required_model_paths()
            if not path.is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing classifier model files:\n"
                + "\n".join(
                    f"  {path}"
                    for path in missing
                )
            )

    def load(self) -> None:
        if self._loaded:
            return

        self._validate_model_files()

        print("Loading V7 models by rebuilding architectures...")

        # --------------------------------------------------
        # TCIC
        # --------------------------------------------------
        for modality in MODALITIES:

            print(
                f"  {modality} TCIC...",
                flush=True,
            )

            model = build_paper_tcic(modality)

            path = (
                self.root
                / f"{modality}_TCIC.keras"
            )

            _load_weights_into_model(
                model,
                path,
            )

            self.tcic[modality] = model

            print(
                f"    OK ({model.count_params():,} params)"
            )

        # --------------------------------------------------
        # TCIE
        # --------------------------------------------------
        for modality in MODALITIES:

            for cls in range(3):

                print(
                    f"  {modality} TCIE class {cls}...",
                    flush=True,
                )

                model = build_paper_tcie(
                    modality,
                    cls,
                )

                path = (
                    self.root
                    / f"{modality}_TCIE_class{cls}.keras"
                )

                _load_weights_into_model(
                    model,
                    path,
                )

                self.tcie[
                    (modality, cls)
                ] = model

                print(
                    f"    OK ({model.count_params():,} params)"
                )

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------
        print(
            "  Fusion ANN...",
            flush=True,
        )

        fusion = build_fusion()

        _load_weights_into_model(
            fusion,
            self.root
            / "fusion_ann_best.keras",
        )

        self.fusion = fusion

        print(
            f"    OK ({fusion.count_params():,} params)"
        )

        self._loaded = True

        print("All 13 V7 models loaded successfully.")

    @staticmethod
    def vmax_to_class(
        vmax: float,
    ) -> tuple[int, str]:

        if not np.isfinite(vmax):
            return 0, CLASS_NAMES[0]

        if vmax < 64.0:
            return 0, CLASS_NAMES[0]

        if vmax < 96.0:
            return 1, CLASS_NAMES[1]

        return 2, CLASS_NAMES[2]

    @staticmethod
    def _validate_modalities(
        images: dict[str, np.ndarray],
    ) -> None:

        missing = sorted(
            set(MODALITIES)
            - set(images)
        )

        if missing:
            raise ValueError(
                f"Missing modalities: {missing}. "
                f"Required: {list(MODALITIES)}"
            )

    def predict(
        self,
        images: dict[str, np.ndarray],
    ) -> dict:

        if not self._loaded:
            raise RuntimeError(
                "V7 classifier is not loaded. "
                "Call load() before predict()."
            )

        self._validate_modalities(images)

        branches = {}

        # ==================================================
        # IR / WV / VIS
        # ==================================================
        for modality in MODALITIES:

            x = validate_model_input(
                images[modality]
            )

            prob = np.asarray(
                self.tcic[modality].predict(
                    x,
                    batch_size=4,
                    verbose=0,
                ),
                dtype=np.float32,
            )

            if (
                prob.ndim != 2
                or prob.shape[0] != len(x)
                or prob.shape[1] != 3
            ):
                raise RuntimeError(
                    f"{modality} TCIC produced "
                    f"unexpected output shape: "
                    f"{prob.shape}; expected "
                    f"({len(x)}, 3)."
                )

            cls = np.argmax(
                prob,
                axis=1,
            ).astype(np.int64)

            vmax = np.empty(
                len(x),
                dtype=np.float32,
            )

            # Route each sample to its predicted
            # TCIE class expert.
            for c in range(3):

                pos = np.where(
                    cls == c
                )[0]

                if len(pos) == 0:
                    continue

                pred = np.asarray(
                    self.tcie[
                        (modality, c)
                    ].predict(
                        x[pos],
                        batch_size=4,
                        verbose=0,
                    ),
                    dtype=np.float32,
                ).reshape(-1)

                if len(pred) != len(pos):
                    raise RuntimeError(
                        f"{modality} TCIE class {c} "
                        f"returned {len(pred)} predictions "
                        f"for {len(pos)} samples."
                    )

                vmax[pos] = pred

            if not np.all(
                np.isfinite(vmax)
            ):
                raise RuntimeError(
                    f"{modality} TCIE routing produced "
                    "non-finite Vmax predictions."
                )

            branches[modality] = {
                "probabilities": prob,
                "class_index": cls,
                "vmax_kt": vmax,
            }

        # ==================================================
        # FUSION
        # ==================================================
        fusion_in = np.column_stack(
            [
                branches["IR"]["vmax_kt"],
                branches["WV"]["vmax_kt"],
                branches["VIS"]["vmax_kt"],
            ]
        ).astype(np.float32)

        if self.fusion is None:
            raise RuntimeError(
                "Fusion model is not loaded."
            )

        final_vmax = np.asarray(
            self.fusion.predict(
                fusion_in,
                batch_size=32,
                verbose=0,
            ),
            dtype=np.float32,
        ).reshape(-1)

        if len(final_vmax) != len(fusion_in):
            raise RuntimeError(
                "Fusion ANN returned an unexpected "
                "number of predictions."
            )

        if not np.all(
            np.isfinite(final_vmax)
        ):
            raise RuntimeError(
                "Fusion ANN produced non-finite "
                "Vmax predictions."
            )

        final_class_index = np.array(
            [
                self.vmax_to_class(
                    float(v)
                )[0]
                for v in final_vmax
            ],
            dtype=np.int64,
        )

        final_class = [
            CLASS_NAMES[int(i)]
            for i in final_class_index
        ]

        return {
            "branches": branches,
            "fusion_inputs_vmax_kt": fusion_in,
            "final_vmax_kt": final_vmax,
            "final_class_index": final_class_index,
            "final_class": final_class,
        }