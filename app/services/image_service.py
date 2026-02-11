import io
import numpy as np
from PIL import Image

TARGET_SIZE = (224, 224)  # standard size for many CNNs

def load_and_verify_image(image_bytes: bytes) -> Image.Image:
    """
    Opens the uploaded bytes as an image and verifies it's not corrupted.
    Returns a PIL Image object.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # verifies file integrity
        # Need to reopen after verify because verify() can invalidate the file pointer
        img = Image.open(io.BytesIO(image_bytes))
        return img
    except Exception as e:
        raise ValueError("Uploaded file is not a valid image") from e

def preprocess_for_tf(img: Image.Image) -> np.ndarray:
    """
    Converts PIL image to a TensorFlow-ready numpy array:
    - RGB
    - resize to 224x224
    - normalize to 0-1
    - add batch dimension (1, 224, 224, 3)
    """
    img = img.convert("RGB")
    img = img.resize(TARGET_SIZE)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr
