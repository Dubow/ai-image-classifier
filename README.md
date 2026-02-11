# 🌿 Plant Disease Detection API

A FastAPI-based backend system for detecting plant leaf diseases using a deep learning model trained on the Kaggle Plant Disease Dataset.

This API allows users to upload a plant leaf image and receive:

- 🌱 Crop name  
- 🟢 Health status (Healthy / Diseased)  
- 🦠 Specific disease (if present)  
- 📊 Confidence score  
- 🔎 Top 3 model predictions  

---

## 🚀 Tech Stack

- **FastAPI** – Backend API framework  
- **TensorFlow / Keras** – Deep learning model  
- **MobileNetV2** – Transfer learning backbone  
- **NumPy & Pillow** – Image preprocessing  
- **Uvicorn** – ASGI server  

---

## 📁 Project Structure

```bash
app/
 ├── api/v1/predict.py        # Prediction endpoint
 ├── services/
 │    ├── image_service.py    # Image preprocessing
 │    ├── model_service.py    # Model loading & inference
 │    └── plant_gate.py       # Plant domain validation
 └── main.py                  # FastAPI entry point

models/
 ├── plant_model.keras        # Trained TensorFlow model
 └── labels.json              # Class labels

requirements.txt
README.md
```


## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd image-analyzer-backend
2️⃣ Create a virtual environment
python -m venv venv
Activate it:

Windows

venv\Scripts\activate
Mac/Linux

source venv/bin/activate
3️⃣ Install dependencies
pip install -r requirements.txt
▶️ Running the Server
Start the development server:

uvicorn app.main:app --reload
The API will be available at:

http://127.0.0.1:8000
Swagger Documentation:

http://127.0.0.1:8000/docs
📡 API Endpoint
POST /api/v1/predict/
Upload a plant leaf image.

Request
Content-Type: multipart/form-data

Field name: file

Example cURL Request
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@leaf.jpg;type=image/jpeg'
Example Response
{
  "filename": "leaf.png",
  "top_label": "Tomato___Late_blight",
  "top_confidence": 0.8421,
  "category": "OtherDisease",
  "crop": "Tomato",
  "status": "Diseased",
  "disease": "Late blight",
  "predictions": [
    {
      "label": "Tomato___Late_blight",
      "confidence": 0.8421
    },
    {
      "label": "Tomato___Early_blight",
      "confidence": 0.0912
    },
    {
      "label": "Potato___Late_blight",
      "confidence": 0.0415
    }
  ]
}
🧠 How the System Works
User uploads an image.

Image is validated and resized to 224x224.

Basic quality checks are applied.

(Optional) Domain gate checks if image appears to be a plant.

The trained CNN model predicts disease probabilities.

Confidence thresholds determine whether result is valid or marked as unknown.

📊 Dataset
Model trained using:

Kaggle Plant Disease Dataset
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

The dataset contains multiple crops and disease classes including:

Tomato

Corn (Maize)

Apple

Grape

Strawberry

Pepper

Potato

Squash

Cherry

And more...

🛡 Confidence Handling
The API marks predictions as unknown when:

Top confidence is below threshold

Margin between top predictions is too small

Image quality is poor

The image does not resemble a plant

This prevents misleading classifications.

⚠️ Limitations
Works best with clear, close-up leaf images.

Background clutter may reduce accuracy.

Not designed for full plant or field-level images.

Not suitable for non-plant objects.

👨‍💻 Author
Abdirahman Dubow
Plant Disease Detection Backend
FastAPI + TensorFlow Project

📌 Future Improvements
Add model versioning

Deploy to cloud (Render / Railway / AWS)

Add database logging

Integrate frontend UI

Improve plant/non-plant detection

📄 License
This project is for academic and research purposes. Not for commercial use. Please credit the author if you use or modify this code.