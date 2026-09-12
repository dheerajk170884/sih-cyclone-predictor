# SIH Cyclone Intensity Classifier — V7

Standalone inference repository for the final validated SIH IR/WV/VIS cyclone-intensity model.

## Pipeline

```text
Raw TCIR-CPAC_IO_SH image
(201 x 201 x 4)
        |
        v
V4 preprocessing
- invalid/non-finite -> channel median
- P01/P99 clipping
- map each channel to [-1, 1]
- bilinear antialiased resize to 170 x 170
        |
        +--> IR -> TCIC -> routed TCIE -> IR Vmax
        +--> WV -> TCIC -> routed TCIE -> WV Vmax
        +--> VIS -> TCIC -> routed TCIE -> VIS Vmax
                                      |
                                      v
                            Fusion ANN (3 -> 8 -> 4 -> 1)
                                      |
                                      v
                               Final Vmax (kt)
                                      |
                                      v
                           SIH intensity class
```

The V7 classifier uses IR, WV and VIS only. PMW is not part of this final model.

## Preprocessing

The repository performs the same reconstructed V4 inference preprocessing used for model development. For each of IR, WV and VIS, invalid/non-finite values and values with absolute magnitude above `1e30` are replaced with the channel median, values are clipped to the robust P01/P99 interval, mapped linearly to `[-1,1]`, then resized from `201x201` to `170x170` with TensorFlow bilinear antialiasing.

The stored statistics are embedded in `sih_classifier/preprocessing.py`, so a separate normalization-statistics file is not required for inference.

## Model files

Place these 13 trained Keras models in `models/`:

```text
fusion_ann_best.keras
IR_TCIC.keras
IR_TCIE_class0.keras
IR_TCIE_class1.keras
IR_TCIE_class2.keras
WV_TCIC.keras
WV_TCIE_class0.keras
WV_TCIE_class1.keras
WV_TCIE_class2.keras
VIS_TCIC.keras
VIS_TCIE_class0.keras
VIS_TCIE_class1.keras
VIS_TCIE_class2.keras
```

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## End-to-end prediction

Input `.npy` must contain raw images with shape `(N,201,201,4)` in channel order `IR, WV, VIS, PMW`.

```bash
python pipeline.py --models models --input sample.npy --output prediction.json
```

The output contains branch-level TCIC probabilities, branch Vmax values, final fused Vmax, class index and class name.

## Python API

```python
from sih_classifier import SIHClassifierPipeline

pipeline = SIHClassifierPipeline("models")
result = pipeline.predict_raw(raw_images)
print(result["final_vmax_kt"])
print(result["final_class"])
```

## Scientific scope

This repository is inference-only. It intentionally does not contain dataset splitting, training augmentation, training callbacks, or model-selection code.
