from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import io

from app.services.model_service import predict_topk

router = APIRouter(prefix="/predict", tags=["Prediction"])

ALLOWED_TYPES = ["image/jpg", "image/jpeg", "image/png"]

# --- Confidence rules (for your 38-class model) ---
TOP1_MIN = 0.55
MARGIN_MIN = 0.10

# --- Image quality gate ---
MIN_STD = 12.0
MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 245.0

# --- Leafness gate (green-ish pixels) ---
# Keep low to allow powdery/gray leaves, but still block obvious screenshots
LEAF_GREEN_RATIO_MIN = 0.04


def parse_label(label: str):
    crop, condition = label.split("___", 1)
    if condition.lower() == "healthy":
        return crop, "Healthy", None
    return crop, "Diseased", condition.replace("_", " ")


def to_category(label: str) -> str:
    if label == "unknown":
        return "Unknown"

    _, condition = label.split("___", 1)
    c = condition.lower()

    if c == "healthy":
        return "Healthy"
    if "powdery" in c and "mildew" in c:
        return "Powdery"
    if "rust" in c:
        return "Rust"
    return "OtherDisease"


def basic_image_quality(arr_224_rgb: np.ndarray):
    gray = arr_224_rgb.mean(axis=2)
    std = float(gray.std())
    mean = float(gray.mean())

    if std < MIN_STD:
        return False, f"Low detail/flat image (std={std:.1f})"
    if mean < MIN_BRIGHTNESS:
        return False, f"Image too dark (mean={mean:.1f})"
    if mean > MAX_BRIGHTNESS:
        return False, f"Image too bright (mean={mean:.1f})"

    return True, None


def leafness_gate(arr_224_rgb: np.ndarray):
    r = arr_224_rgb[..., 0]
    g = arr_224_rgb[..., 1]
    b = arr_224_rgb[..., 2]

    green_mask = (g > 60) & (g > r + 10) & (g > b + 10)
    green_ratio = float(green_mask.mean())

    return green_ratio >= LEAF_GREEN_RATIO_MIN, green_ratio


@router.post("/")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG and PNG images are allowed")

    image_bytes = await file.read()

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)  # 0..255

    # 1) Quality check
    ok, reason = basic_image_quality(arr)
    if not ok:
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "top_label": "unknown",
            "top_confidence": None,
            "category": "Unknown",
            "crop": None,
            "status": None,
            "disease": None,
            "predictions": [],
            "note": f"Unclear image: {reason}. Upload a clear close-up leaf photo.",
        }

    # 2) Leafness gate
    is_leaf, green_ratio = leafness_gate(arr)
    if not is_leaf:
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "top_label": "unknown",
            "top_confidence": None,
            "category": "Unknown",
            "crop": None,
            "status": None,
            "disease": None,
            "predictions": [],
            "note": "Not a leaf image (low green content). Upload a clear leaf photo.",
            "debug": {
                "green_ratio": round(green_ratio, 4),
                "LEAF_GREEN_RATIO_MIN": LEAF_GREEN_RATIO_MIN,
            },
        }

    # 3) Predict
    batch = np.expand_dims(arr, axis=0)  # (1,224,224,3)
    preds = predict_topk(batch, top_k=3)

    top1 = float(preds[0]["confidence"])
    top2 = float(preds[1]["confidence"]) if len(preds) > 1 else 0.0
    margin = top1 - top2

    low_conf = (top1 < TOP1_MIN) or (margin < MARGIN_MIN)

    top_label = "unknown" if low_conf else preds[0]["label"]
    category = to_category(top_label)

    crop = status = disease = None
    if top_label != "unknown":
        crop, status, disease = parse_label(top_label)

    preds_clean = [{"label": p["label"], "confidence": round(float(p["confidence"]), 4)} for p in preds]

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "top_label": top_label,
        "top_confidence": None if low_conf else round(top1, 4),

        "category": category,
        "crop": crop,
        "status": status,
        "disease": disease,

        "predictions": preds_clean,
        "note": "Low confidence prediction. Try a clearer close-up leaf photo." if low_conf else None,
        "debug": {
            "green_ratio": round(green_ratio, 4),
            "top1": round(top1, 4),
            "top2": round(top2, 4),
            "margin": round(margin, 4),
            "TOP1_MIN": TOP1_MIN,
            "MARGIN_MIN": MARGIN_MIN,
        },
    }
