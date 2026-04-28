
# 🍅 Tomato Leaf Disease Prediction (2026 Edition)

> **🌍 Experience the AI:** <a href="https://huggingface.co/spaces/saransh619/Tomato-Disease-AI" target="_blank">Click here to view and interact with the Live Demo of this project.</a>

A professional, deep-learning-powered web application that identifies 10 different types of tomato plant diseases using **MobileNetV2 Transfer Learning**. 

> **🔄 Project Evolution:** Originally developed in 2022 as my Final Year University Project using a basic CNN, I have completely re-architected and modernized the system in 2026. The most significant upgrade was migrating from a simple CNN to **MobileNetV2 Transfer Learning**, which increased real-world accuracy and made the model robust against background noise and non-leaf images.

## 🌟 Key Features
- **Modern Dark Mode UI**: A stunning, responsive dashboard built with 2026 design principles.
- **Real-time Prediction**: Instant analysis of tomato leaves with confidence scoring.
- **Treatment Suggestions**: Provides medical advice for each identified disease.
- **Cross-Platform**: Fully compatible with Windows, macOS (Intel/M-Series), and Linux.

---

## 🧠 The AI Model (MobileNetV2 Transfer Learning)
This project utilizes **Transfer Learning** with the **MobileNetV2** architecture, pre-trained on the ImageNet dataset. This approach provides significantly higher accuracy and robustness compared to basic CNN models:

1. **Base Model**: MobileNetV2 (frozen) for high-level feature extraction.
2. **Input Layer**: Processes images at `128x128` resolution.
3. **Global Average Pooling**: Reduces the spatial dimensions of the feature maps efficiently.
4. **Custom Head**:
   - Dropout layer (0.3) to prevent overfitting.
   - Dense hidden layer (128 neurons, ReLU activation).
   - Final Dropout layer (0.3).
   - Output layer with 10 neurons (Softmax activation) representing the 10 disease categories.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone git@github.com:saransh619/Tomato-Leaf-Disease-Prediction.git
cd Tomato-Leaf-Disease-Prediction
```

### 2. Set Up a Virtual Environment
**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:8081`.

---

## ☁️ Cloud Deployment (Docker & Hugging Face)
This project is configured for seamless deployment on **Hugging Face Spaces** using Docker.

1. The included `Dockerfile` automatically builds the environment.
2. The Flask application dynamically adapts to Hugging Face's required port (`7860`).
3. **To deploy or update the live website**, simply run the deployment script:
   ```bash
   python deploy.py
   ```
   This will securely prompt for your Hugging Face API token and push the code directly, bypassing any Git LFS restrictions. *(Note: If you modify `app.py` or the UI in the future, running this script is the only step required to sync your updates to the live Hugging Face server!)*

---

## 📂 Project Structure
- `app.py`: The main Flask web server.
- `deploy.py`: Secure deployment script for Hugging Face Spaces.
- `model.h5`: The pre-trained AI brain.
- `Training.py`: Script to retrain the model from scratch.
- `Example.py`: A quick terminal-based prediction script.
- `Dataset/`: Organized training, validation, and testing images.

## 🧪 Testing the Model
You can run a quick terminal test without starting the web server:
```bash
python Example.py
```

---

*Refactored with ❤️ in 2026 for a high-performance, modern user experience.*
