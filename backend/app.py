import os
import sys
from io import BytesIO

import numpy as np

from flask import Flask, jsonify, request, render_template

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFIER_DIR = os.path.join(BASE_DIR, "ml", "sih-cyclone-classifier")
sys.path.insert(0, CLASSIFIER_DIR)

from sih_classifier import SIHClassifierPipeline


CLASSIFIER_MODELS_DIR = os.path.join(CLASSIFIER_DIR, "models")
DEMO_DATA_PATH = os.path.join(CLASSIFIER_DIR, "demo_data", "demo_sample.npy")
classifier_pipeline = SIHClassifierPipeline(CLASSIFIER_MODELS_DIR)

# Flask automatically routes to the 'static' folder by default
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

@app.get("/")
def index():
    # render_template automatically searches the 'templates' folder
    return render_template("ui.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/api/predict")
def predict():
    data_file = request.files.get("data")
    if data_file is None or not data_file.filename:
        return jsonify({"error": "A raw .npy satellite data file is required."}), 400

    if not data_file.filename.lower().endswith(".npy"):
        return jsonify({"error": "The classifier requires a .npy file."}), 400

    try:
        raw_images = np.load(BytesIO(data_file.read()), allow_pickle=False)
        if raw_images.ndim not in (3, 4) or raw_images.shape[-3:] != (201, 201, 4):
            return jsonify({
                "error": "Expected .npy data shaped (201, 201, 4) or (N, 201, 201, 4)."
            }), 400

        result = classifier_pipeline.predict_raw(raw_images)
        return jsonify({
            "final_vmax_kt": result["final_vmax_kt"].tolist(),
            "final_class_index": result["final_class_index"].tolist(),
            "final_class": result["final_class"],
        })
    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(error)}), 500


@app.post("/api/demo-predict")
def demo_predict():
    try:
        raw_images = np.load(DEMO_DATA_PATH, allow_pickle=False)
        result = classifier_pipeline.predict_raw(raw_images)
        wind_speed = float(result["final_vmax_kt"][0])

        return jsonify({
            "final_vmax_kt": result["final_vmax_kt"].tolist(),
            "final_class_index": result["final_class_index"].tolist(),
            "final_class": result["final_class"],
            "branch_vmax_kt": {
                modality: float(branch["vmax_kt"][0])
                for modality, branch in result["branches"].items()
            },
            "estimated_pressure_drop_hpa": round(max(0.0, wind_speed * 0.8), 1),
        })
    except Exception as error:
        app.logger.exception("Demo prediction failed")
        return jsonify({"error": str(error)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)