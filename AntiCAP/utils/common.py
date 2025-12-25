import os
import sys
import io
import base64
import cv2
import numpy as np
from PIL import Image

def get_model_path(filename: str) -> str:
    """Get absolute path to a model file."""
    if getattr(sys, "frozen", False):
        # If the application is frozen (e.g., packaged with PyInstaller), use the executable directory directly
        base_dir = os.path.dirname(sys.executable)
    else:
        # In development, use the package root (one level above this file's directory)
        base_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base_dir, 'AntiCAP-Models', filename)

def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
            
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        image.load()
        return image
    except Exception as e:
        # Raise ValueError with more context for easier debugging
        raise ValueError(f"Failed to decode base64 image: {e}")

def decode_base64_to_cv2(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image (numpy array)."""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
            
        image_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None

def resize_with_padding(image, target_size, fill_color=(255, 255, 255)):
    target_w, target_h = target_size
    
    # Calculate aspect ratios
    ratio_w = target_w / image.width
    ratio_h = target_h / image.height

    ratio = min(ratio_w, ratio_h)
    
    new_w = int(image.width * ratio)
    new_h = int(image.height * ratio)
    
    # Resize the image
    resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create a new image with the target size and fill color
    new_image = Image.new("RGB", target_size, fill_color)
    new_image.paste(resized_image, (0, 0))
    
    return new_image

def decode(preds, char_map_inv):
    if preds.shape[1] == 1: # Batch size is 1 at dim 1 implies [T, B, C]
         preds = np.transpose(preds, (1, 0, 2)) # [B, T, C]
    
    # Argmax
    preds_idx = np.argmax(preds, axis=2) # [B, T]
    
    decoded_strings = []
    for sequence in preds_idx:
        decoded = []
        prev_char = -1
        for char_idx in sequence:
            if char_idx != 0 and char_idx != prev_char:
                decoded.append(char_map_inv[char_idx])
            prev_char = char_idx
        decoded_strings.append(''.join(decoded))
    return decoded_strings
