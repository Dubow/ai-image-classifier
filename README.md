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

Follow these steps to set up the backend locally.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Dubow/ai-image-classifier.git
cd ai-image-classifier
```

### 2️⃣ Create a Virtual Environment

Make sure you are using Python 3.10 or higher.

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> Ensure that `requirements.txt` contains only the necessary packages (FastAPI, Uvicorn, TensorFlow, Pillow, NumPy, etc.).

---

## ▶️ Running the API

Start the development server:

```bash
uvicorn app.main:app --reload
```

The server will run at:

```
http://127.0.0.1:8000
```

---

## 🌐 API Documentation

### Swagger UI (Interactive Testing)

```
http://127.0.0.1:8000/docs
```

### ReDoc

```
http://127.0.0.1:8000/redoc
```

---

## 📡 Prediction Endpoint

### POST `/api/v1/predict/`

Upload a plant leaf image for analysis.

### 📤 Request Details

- Method: `POST`
- Content-Type: `multipart/form-data`
- Form field: `file`

### Example cURL Request

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@leaf_image.jpg;type=image/jpeg"
```

---

## 📦 Required Model Files

After cloning, ensure the following files exist:

```
models/
├── plant_model.keras
└── labels.json
```

> ⚠️ Model files are not included in the repository due to size limitations.  
Place your trained model inside the `models/` directory before running the server.

---

## 🧠 How the System Works

1. User uploads a leaf image.
2. Image is validated and resized to 224x224.
3. Basic quality checks are applied.
4. A lightweight plant-domain gate checks if the image resembles a plant.
5. The trained TensorFlow model predicts disease probabilities.
6. Confidence thresholds determine whether the result is valid or marked as `unknown`.

---

## 📊 Example Response

```json
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
```

---

## ⚠️ Notes

- Best results are obtained using clear, close-up leaf images.
- Images with heavy background clutter may reduce accuracy.
- Non-leaf images will return `"unknown"`.
- Low confidence predictions include a helpful note in the response.

---

## 🤝 Contributing

If contributing:

```bash
git checkout -b feature/your-feature-name
```

Commit and push changes, then open a Pull Request.

---

## 📄 License

This project is developed for academic and research purposes.  
It may be reused and modified for learning and internship demonstrations.
