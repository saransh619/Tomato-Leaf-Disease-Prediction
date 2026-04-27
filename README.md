# 🍅 Tomato Leaf Disease Prediction (2026 Edition)

A professional, deep-learning-powered web application that identifies 10 different types of tomato plant diseases using a Convolutional Neural Network (CNN). Originally developed in 2022, this project has been fully modernized in 2026 with a premium glassmorphic UI and optimized performance.

## 🌟 Key Features
- **Modern Dark Mode UI**: A stunning, responsive dashboard built with 2026 design principles.
- **Real-time Prediction**: Instant analysis of tomato leaves with confidence scoring.
- **Treatment Suggestions**: Provides medical advice for each identified disease.
- **Cross-Platform**: Fully compatible with Windows, macOS (Intel/M-Series), and Linux.

---

## 🧠 The AI Model (CNN Architecture)
This project uses a custom-trained **Convolutional Neural Network (CNN)** built with TensorFlow/Keras. The architecture is designed to capture fine-grained patterns in leaf textures:

1. **Input Layer**: Processes images at `128x128` resolution.
2. **Convolutional Layers (Conv2D)**: Two layers with 32 filters each to detect features like spots, yellowing, and mold.
3. **Dropout Layers (0.5)**: Integrated after each conv layer to prevent overfitting and ensure high generalization.
4. **MaxPooling**: Reduces spatial dimensions while retaining critical information.
5. **Flattening**: Converts the 2D feature maps into a 1D vector.
6. **Dense (Fully Connected) Layers**:
   - A hidden layer with 128 neurons (ReLU activation).
   - An output layer with 10 neurons (Softmax activation) representing the 10 disease categories.

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
3. To deploy, simply create a new **Docker** space on Hugging Face and push this repository.

---

## 📂 Project Structure
- `app.py`: The main Flask web server.
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
