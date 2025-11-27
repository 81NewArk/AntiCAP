import os
import io
import base64
import re
import numpy as np
import torch
from PIL import Image

from ..utils.common import get_model_path, decode_base64_to_image

def solve_math(manager, img_base64: str, math_model_path: str = '', use_gpu: bool = False):
    math_model_path = math_model_path or get_model_path('[AntiCAP]-CRNN_Math.onnx')

    session = manager.get_onnx_session(math_model_path, use_gpu)
    input_name = session.get_inputs()[0].name

    IMG_H = 32
    IMG_W = 160
    image = decode_base64_to_image(img_base64).convert('RGB')
    image = image.resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)

    image_np = np.array(image, dtype=np.float32) / 255.0
    image_np = (image_np.transpose(2, 0, 1) - 0.5) / 0.5
    input_data = np.expand_dims(image_np, axis=0)

    results = session.run(None, {input_name: input_data})

    CHARS = "0123456789+-*/÷×=?"
    preds_tensor = torch.from_numpy(results[0])
    char_map_inv = {i + 1: c for i, c in enumerate(CHARS)}
    char_map_inv[0] = ' '  # Blank
    
    preds = preds_tensor.permute(1, 0, 2)
    preds = preds.argmax(2)

    decoded_strings = []
    for sequence in preds:
        decoded_chars = []
        prev_char_idx = -1
        for char_idx in sequence:
            char_idx = char_idx.item()
            if char_idx != 0 and char_idx != prev_char_idx:
                decoded_chars.append(char_map_inv[char_idx])
            prev_char_idx = char_idx
        decoded_strings.append(''.join(decoded_chars))
    
    captcha_text = decoded_strings[0] if decoded_strings else ""

    if not captcha_text:
        return None

    # Standardize symbols
    expr = captcha_text
    expr = expr.replace('×', '*').replace('÷', '/')
    expr = expr.replace('？', '?')  # Tolerance for Chinese question mark
    expr = expr.replace('=', '')   # Remove equals sign

    # Remove all non-digit and non-operator characters (question mark will be removed)
    expr = re.sub(r'[^0-9\+\-\*/]', '', expr)

    if not expr:
        return None

    # Safe evaluation
    try:
        result = eval(expr, {"__builtins__": None}, {})
        return result
    except Exception as e:
        print(f"[AntiCAP] 表达式解析出错: {expr}, 错误: {e}")
        return None
