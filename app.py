import os
import uuid
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Create flask instance
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/upload/'

# Load the model once at startup (Perfect for performance)
filepath = './model.h5'
model = load_model(filepath)
print("✅ Tomato Disease AI Loaded")

# Load Security Guard AI (ImageNet)
security_model = MobileNetV2(weights='imagenet')
print("🛡️ Security Guard AI Loaded")

# Disease Information Dictionary (The Source of Truth)
DISEASE_INFO = {
    0: {"title": "Tomato - Bacterial Spot Disease", "treatment": "Use copper-based fungicides and avoid overhead watering to prevent spread."},
    1: {"title": "Tomato - Early Blight Disease", "treatment": "Remove infected lower leaves and apply fungicides containing chlorothalonil or copper."},
    2: {"title": "Tomato - Healthy and Fresh", "treatment": "Your plant is doing great! Keep up the good work with regular watering and sunlight."},
    3: {"title": "Tomato - Late Blight Disease", "treatment": "Extremely dangerous! Remove infected plants immediately. Apply copper-based sprays to others."},
    4: {"title": "Tomato - Leaf Mold Disease", "treatment": "Improve air circulation around plants and reduce humidity. Use calcium-based sprays if needed."},
    5: {"title": "Tomato - Septoria Leaf Spot Disease", "treatment": "Remove infected leaves. Apply fungicides like chlorothalonil or copper-based sprays."},
    6: {"title": "Tomato - Target Spot Disease", "treatment": "Avoid overhead irrigation. Apply fungicides containing azoxystrobin or chlorothalonil."},
    7: {"title": "Tomato - Tomato Yellow Leaf Curl Virus Disease", "treatment": "Spread by whiteflies. Use insecticidal soaps and remove infected plants immediately."},
    8: {"title": "Tomato - Tomato Mosaic Virus Disease", "treatment": "No cure. Remove and burn infected plants. Wash hands and tools thoroughly."},
    9: {"title": "Tomato - Two-Spotted Spider Mite Disease", "treatment": "Use neem oil or insecticidal soaps. Increasing humidity can also help."},
}

def is_plant_image(image_path):
    """
    Uses Google's ImageNet model to verify if the image is actually a plant.
    Returns (True, label) if it's a plant, (False, label) otherwise.
    """
    try:
        # ImageNet expects 224x224 input
        img = load_img(image_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x) # Correct scaling for ImageNet
        
        preds = security_model.predict(x, verbose=0)
        decoded = decode_predictions(preds, top=3)[0]
        
        # Keywords that suggest it IS a plant or natural environment
        plant_keywords = ['leaf', 'plant', 'tree', 'vegetable', 'fruit', 'green', 'pepper', 'corn', 'daisy', 'grass', 'pot', 'garden']
        
        for imagenet_id, label, score in decoded:
            label_lower = label.lower()
            if any(key in label_lower for key in plant_keywords):
                return True, label
                
        return False, decoded[0][1]
    except Exception as e:
        print(f"❌ Security check error: {e}")
        return True, "unknown"

def pred_tomato_disease(image_path):
    """
    Handles image preprocessing and model prediction.
    Target size 128x128 matches the training notebook configuration.
    """
    try:
        test_image = load_img(image_path, target_size=(128, 128))
        test_image = img_to_array(test_image) / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        
        result = model.predict(test_image)
        pred_idx = np.argmax(result, axis=1)[0]
        confidence = float(np.max(result) * 100)
        
        return pred_idx, confidence
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None, 0

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400
        
    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    # PERFECT PERFORMANCE: Give file a unique name to avoid cache issues
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)

    # 🛡️ STEP 1: Security Guard Check
    is_valid, detected_label = is_plant_image(file_path)
    if not is_valid:
        return render_template('index.html', 
                               prediction="Invalid Image Detected", 
                               treatment=f"The AI thinks this is a '{detected_label.replace('_', ' ')}', not a plant. Please upload a clear photo of a tomato leaf.", 
                               confidence="0.00",
                               user_image=file_path)

    # 🧬 STEP 2: Tomato Disease Prediction
    pred_idx, confidence = pred_tomato_disease(file_path)
    
    if pred_idx is None:
        return "Error processing image", 500

    # BRAVE LOGIC: Lowered to 45% because the Security Guard (is_plant_image) 
    # handles non-plant rejection already!
    if confidence < 45.0:
        return render_template('index.html', 
                               prediction="Unrecognized Leaf Pattern", 
                               treatment="The AI sees a plant, but the pattern is too blurry or far away to be sure. Please try a closer photo of a single leaf.", 
                               confidence=f"{confidence:.2f}",
                               user_image=file_path)
    
    # Get info from our dictionary
    info = DISEASE_INFO.get(pred_idx, {"title": "Unknown", "treatment": "Consult an expert."})
    
    return render_template('index.html', 
                           prediction=info['title'], 
                           treatment=info['treatment'], 
                           confidence=f"{confidence:.2f}",
                           user_image=file_path)

import os

if __name__ == "__main__":
    # Use environment PORT if available (e.g. 7860 in Docker), otherwise default to 8081 locally
    port = int(os.environ.get("PORT", 8081))
    app.run(host='0.0.0.0', port=port, threaded=False, debug=False)