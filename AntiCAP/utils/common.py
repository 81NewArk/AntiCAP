import os
import io
import base64
import cv2
import numpy as np
from PIL import Image

def get_model_path(filename: str) -> str:
    """Get absolute path to a model file."""
    current_dir = os.path.dirname(__file__)
    # Go up one level to AntiCAP root
    package_root = os.path.dirname(current_dir)
    return os.path.join(package_root, 'AntiCAP-Models', filename)

def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")

def decode_base64_to_cv2(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image (numpy array)."""
    try:
        image_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None
