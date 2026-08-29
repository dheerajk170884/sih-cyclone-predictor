import io
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm
from flask import Flask, request, jsonify

# IMD Cyclone Categories
CATEGORIES = [
    "Depression (31-49 km/h)",
    "Deep Depression (50-61 km/h)",
    "Cyclonic Storm (62-88 km/h)",
    "Severe Cyclonic Storm (89-117 km/h)",
    "Very Severe Cyclonic Storm (118-166 km/h)",
    "Super Cyclonic Storm (>=222 km/h)"
]

app = Flask(__name__)

# Model Definition
class CycloneIntensityClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(CycloneIntensityClassifier, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CycloneIntensityClassifier(num_classes=6).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ML Cyclone Prediction Service Active (Flask)"})

@app.route("/predict", methods=["POST"])
def predict_cyclone():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class_id = torch.argmax(probabilities).item()
        confidence = float(probabilities[predicted_class_id].item())
        
    return jsonify({
        "predicted_category": CATEGORIES[predicted_class_id],
        "category_id": predicted_class_id,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)