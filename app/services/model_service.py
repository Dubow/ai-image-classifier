# app/services/model_service.py
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "plant_model.keras"
LABELS_PATH = BASE_DIR / "models" / "labels.json"

_model = None
_labels = None

def get_model_and_labels():
    global _model, _labels
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if _labels is None:
        _labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return _model, _labels

def predict_topk(img_batch: np.ndarray, top_k: int = 3):
    """
    img_batch: (1,224,224,3) float32, either 0..255 or 0..1
    """
    model, labels = get_model_and_labels()

    x = img_batch.astype("float32")
    # your model has mobilenet_v2 preprocess inside, so feed 0..255
    if x.max() <= 1.0:
        x = x * 255.0

    probs = model.predict(x, verbose=0)[0]
    idx = np.argsort(probs)[::-1][:top_k]

    return [{"label": labels[i], "confidence": float(probs[i])} for i in idx]
