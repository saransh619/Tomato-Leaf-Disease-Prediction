import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Use relative paths so it works on any computer (Mac or Windows)
filepath = './model.h5'
model = load_model(filepath)
print("✅ AI Model Loaded Successfully")

# Disease Dictionary for mapping results to names
DISEASE_INFO = {
    0: "Tomato - Bacterial Spot Disease",
    1: "Tomato - Early Blight Disease",
    2: "Tomato - Healthy and Fresh",
    3: "Tomato - Late Blight Disease",
    4: "Tomato - Leaf Mold Disease",
    5: "Tomato - Septoria Leaf Spot Disease",
    6: "Tomato - Target Spot Disease",
    7: "Tomato - Tomato Yellow Leaf Curl Virus Disease",
    8: "Tomato - Tomato Mosaic Virus Disease",
    9: "Tomato - Two-Spotted Spider Mite Disease",
}

# Pick a test image from your dataset
# We use a relative path here so it works on your Mac
image_path = './Dataset/Testing/Early_Blight.JPG'

if not os.path.exists(image_path):
    print(f"❌ Error: Could not find image at {image_path}")
else:
    # Load and preprocess
    test_image = load_img(image_path, target_size=(128, 128))
    test_image = img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    # Predict
    result = model.predict(test_image)
    pred_idx = np.argmax(result, axis=1)[0]
    confidence = float(np.max(result) * 100)

    # Output result
    info = DISEASE_INFO.get(pred_idx, {"title": "Unknown", "treatment": "Consult an expert."})
    
    print("-" * 30)
    print(f"🔍 PREDICTION RESULT:")
    print(f"Disease Name: {info['title']}")
    print(f"Confidence:   {confidence:.2f}%")
    print(f"Treatment:    {info['treatment']}")
    print("-" * 30)
