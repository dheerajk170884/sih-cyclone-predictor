import io
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Model Paths
BASE_DIR = os.path.dirname(__file__)
IDENTIFIER_PATH = os.path.normpath(os.path.join(BASE_DIR, "../models/cyclone_identifier.keras"))
CLASSIFIER_PATH = os.path.normpath(os.path.join(BASE_DIR, "../models/cyclone_efficientnet_b0.keras"))

CATEGORIES = [
    "Depression", 
    "Deep Depression", 
    "Cyclonic Storm",
    "Severe Cyclonic Storm", 
    "Very Severe Cyclonic Storm", 
    "Super Cyclonic Storm"
]

_identifier_model = None
_classifier_model = None

def load_models():
    global _identifier_model, _classifier_model
    if _identifier_model is None:
        if not os.path.exists(IDENTIFIER_PATH):
            raise FileNotFoundError(f"Identifier model missing at {IDENTIFIER_PATH}")
        _identifier_model = tf.keras.models.load_model(IDENTIFIER_PATH)
        
    if _classifier_model is None:
        if os.path.exists(CLASSIFIER_PATH):
            _classifier_model = tf.keras.models.load_model(CLASSIFIER_PATH)
            
    return _identifier_model, _classifier_model

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)

def predict_cyclone_intensity(image_bytes: bytes) -> dict:
    identifier, classifier = load_models()
    processed_tensor = preprocess_image(image_bytes)

    # 1. TERA IDENTIFIER CHECK (Presence Detection)
    presence_pred = float(identifier.predict(processed_tensor, verbose=0)[0][0])
    
    # 0 = non_cyclone, 1 = cyclone
    if presence_pred < 0.5:
        return {
            "is_cyclone": False,
            "predicted_category": "No Cyclone Detected",
            "category_id": -1,
            "confidence": round(float(1.0 - presence_pred), 4)
        }

    # 2. FRIEND'S CLASSIFIER (If Cyclone Detected)
    if classifier is None:
        return {
            "is_cyclone": True,
            "predicted_category": "Cyclone Detected (Classifier Model Training Pending)",
            "category_id": 0,
            "confidence": round(presence_pred, 4)
        }

    predictions = classifier.predict(processed_tensor, verbose=0)[0]
    predicted_class_id = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class_id])

    return {
        "is_cyclone": True,
        "predicted_category": CATEGORIES[predicted_class_id],
        "category_id": predicted_class_id,
        "confidence": round(confidence, 4)
    }