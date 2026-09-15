import os
import sys
import csv
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image

from flask import Flask, jsonify, request, render_template

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFIER_DIR = os.path.join(BASE_DIR, "ml", "sih-cyclone-classifier")
sys.path.insert(0, CLASSIFIER_DIR)

from sih_classifier import SIHClassifierPipeline


CLASSIFIER_MODELS_DIR = os.path.join(CLASSIFIER_DIR, "models")
DEMO_DATA_PATH = os.path.join(CLASSIFIER_DIR, "demo_data", "demo_sample.npy")
IDENTIFIER_MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "cyclone_identifier.keras")
IDENTIFIER_NON_CYCLONE_THRESHOLD = 0.05
IMPACT_RADIUS_MIN_KM = 100.0
IMPACT_RADIUS_MAX_KM = 300.0
classifier_pipeline = SIHClassifierPipeline(CLASSIFIER_MODELS_DIR)
identifier_model = None

# Flask automatically routes to the 'static' folder by default
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def _load_identifier_model():
    global identifier_model
    if identifier_model is None:
        if not os.path.isfile(IDENTIFIER_MODEL_PATH):
            raise FileNotFoundError(
                f"Identifier model missing at {IDENTIFIER_MODEL_PATH}"
            )
        identifier_model = tf.keras.models.load_model(IDENTIFIER_MODEL_PATH)
    return identifier_model


def _read_satellite_image(image_file) -> np.ndarray:
    if image_file is None or not image_file.filename:
        raise ValueError("All three satellite images are required.")

    try:
        image = Image.open(image_file.stream).convert("L")
        image = image.resize((201, 201), Image.Resampling.BILINEAR)
    except Exception as error:
        raise ValueError(f"Could not read {image_file.filename}: {error}") from error

    return np.asarray(image, dtype=np.float32)


def _identifier_prediction(
    visible_file,
    infrared_file,
    water_vapour_file,
) -> tuple[bool, float]:
    for image_file in (visible_file, infrared_file, water_vapour_file):
        if image_file is None or not image_file.filename:
            raise ValueError("All three satellite images are required.")

    visible_file.stream.seek(0)
    image = Image.open(visible_file.stream).convert("RGB").resize(
        (224, 224), Image.Resampling.BILINEAR
    )
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.keras.applications.efficientnet.preprocess_input(
        np.expand_dims(image_array, axis=0)
    )
    probability = float(_load_identifier_model().predict(image_array, verbose=0)[0][0])
    # The training dataset maps the first alphabetical class (cyclone) to 0.
    # The sigmoid is therefore the non-cyclone probability.
    return probability < IDENTIFIER_NON_CYCLONE_THRESHOLD, probability


def _overwrite_demo_data(raw_images: np.ndarray) -> None:
    os.makedirs(os.path.dirname(DEMO_DATA_PATH), exist_ok=True)
    temporary_path = f"{DEMO_DATA_PATH}.tmp"
    with open(temporary_path, "wb") as data_file:
        np.save(data_file, raw_images, allow_pickle=False)
        data_file.flush()
        os.fsync(data_file.fileno())
    os.replace(temporary_path, DEMO_DATA_PATH)


def _read_track_csv(track_file) -> list[dict]:
    if track_file is None or not track_file.filename:
        raise ValueError("A cyclone track CSV file is required.")
    if not track_file.filename.lower().endswith(".csv"):
        raise ValueError("The cyclone track must be a .csv file.")

    track_file.stream.seek(0)
    try:
        text = track_file.stream.read().decode("utf-8-sig")
        reader = csv.DictReader(text.splitlines())
        if not reader.fieldnames:
            raise ValueError("The CSV has no header row.")

        columns = {
            name.strip().lower(): name
            for name in reader.fieldnames
            if name and name.strip()
        }
        required = {"lon", "lat", "time"}
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(
                "The CSV must contain these columns: lon, lat, time. "
                f"Missing: {', '.join(missing)}."
            )

        track = []
        for row_number, row in enumerate(reader, start=2):
            try:
                longitude = float(row[columns["lon"]])
                latitude = float(row[columns["lat"]])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid lon/lat values on CSV row {row_number}."
                ) from error

            if not np.isfinite(longitude) or not 0 <= longitude <= 360:
                raise ValueError(
                    f"Longitude on CSV row {row_number} must be between 0 and 360."
                )
            if not np.isfinite(latitude) or not -90 <= latitude <= 90:
                raise ValueError(
                    f"Latitude on CSV row {row_number} must be between -90 and 90."
                )

            point = {
                "longitude": longitude,
                "longitude_normalized": longitude - 360 if longitude > 180 else longitude,
                "latitude": latitude,
                "time": str(row[columns["time"]]).strip(),
            }
            for column in ("data_set", "id"):
                if column in columns:
                    point[column] = str(row[columns[column]]).strip()
            track.append(point)
    except UnicodeDecodeError as error:
        raise ValueError("The CSV must be UTF-8 encoded.") from error

    if not track:
        raise ValueError("The cyclone track CSV has no data rows.")
    return track


def _prediction_response(raw_images: np.ndarray, track: list[dict]) -> dict:
    result = classifier_pipeline.predict_raw(raw_images)
    wind_speed = float(result["final_vmax_kt"][0])
    current_position = track[-1]
    impact_radius_km = min(
        IMPACT_RADIUS_MAX_KM,
        max(IMPACT_RADIUS_MIN_KM, 80.0 + 1.5 * wind_speed),
    )
    return {
        "final_vmax_kt": result["final_vmax_kt"].tolist(),
        "final_class_index": result["final_class_index"].tolist(),
        "final_class": result["final_class"],
        "branch_vmax_kt": {
            modality: float(branch["vmax_kt"][0])
            for modality, branch in result["branches"].items()
        },
        "estimated_pressure_drop_hpa": round(max(0.0, wind_speed * 0.8), 1),
        "impact_radius_km": round(impact_radius_km, 1),
        "impact_area_km2": round(np.pi * impact_radius_km ** 2, 1),
        "longitude": current_position["longitude"],
        "latitude": current_position["latitude"],
        "track": track,
    }

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


@app.post("/api/run-predict")
@app.post("/api/upload-predict")
def upload_predict():
    try:
        track = _read_track_csv(request.files.get("track"))
        visible_file = request.files.get("visible")
        infrared_file = request.files.get("infrared")
        water_vapour_file = request.files.get("water_vapour")
        is_cyclone, identifier_probability = _identifier_prediction(
            visible_file,
            infrared_file,
            water_vapour_file,
        )

        if not is_cyclone:
            return jsonify({
                "is_cyclone": False,
                "identifier_confidence": round(identifier_probability, 4),
                "message": "No cyclone detected in the visible image.",
            })

        raw_images = np.stack([
            _read_satellite_image(infrared_file),
            _read_satellite_image(water_vapour_file),
            _read_satellite_image(visible_file),
            np.zeros((201, 201), dtype=np.float32),
        ], axis=-1)
        response = _prediction_response(raw_images, track)
        _overwrite_demo_data(raw_images)
        response.update({
            "is_cyclone": True,
            "identifier_confidence": round(1.0 - identifier_probability, 4),
            "message": "Cyclone detected. Intensity classified from fused Vmax.",
        })
        return jsonify(response)
    except Exception as error:
        app.logger.exception("Uploaded prediction failed")
        return jsonify({"error": str(error)}), 400


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