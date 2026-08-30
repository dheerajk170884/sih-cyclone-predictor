import io
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# IMD Cyclone Categories
CATEGORIES = [
    "Depression (31-49 km/h)",
    "Deep Depression (50-61 km/h)",
    "Cyclonic Storm (62-88 km/h)",
    "Severe Cyclonic Storm (89-117 km/h)",
    "Very Severe Cyclonic Storm (118-166 km/h)",
    "Super Cyclonic Storm (>=222 km/h)"
]

# Path to trained model weights inside ml/models/
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'cyclone_model_baseline.keras')

_model = None

def load_cyclone_model():
    """Model ko ek baar memory mein load karega (Lazy Loading)."""
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = tf.keras.models.load_model(MODEL_PATH)
        else:
            # Fallback: Agar trained weights nahi mile toh EfficientNet initialize karega
            base_model = tf.keras.applications.EfficientNetB0(
                weights='imagenet', 
                include_top=False, 
                input_shape=(224, 224, 3)
            )
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            preds = tf.keras.layers.Dense(len(CATEGORIES), activation='softmax')(x)
            _model = tf.keras.models.Model(inputs=base_model.input, outputs=preds)
    return _model

def preprocess_image(image_bytes):
    """Raw image bytes ko model ke input format mein convert karega."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)

def predict_cyclone_intensity(image_bytes: bytes) -> dict:
    """
    Main function jise Backend (app.py) call karega.
    Input: image bytes
    Output: Dictionary with category, category_id, aur confidence
    """
    model = load_cyclone_model()
    processed_tensor = preprocess_image(image_bytes)

    predictions = model.predict(processed_tensor)
    predicted_class_id = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    return {
        "predicted_category": CATEGORIES[predicted_class_id],
        "category_id": predicted_class_id,
        "confidence": round(confidence * 100, 2)
    }

if __name__ == "__main__":
    print("Testing ML Inference pipeline...")
    # Dummy black image create karke test inference run karte hain
    dummy_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    buffer = io.BytesIO()
    dummy_img.save(buffer, format="JPEG")
    
    output = predict_cyclone_intensity(buffer.getvalue())
    print("Test Successful! Sample Prediction Output:")
    print(output)