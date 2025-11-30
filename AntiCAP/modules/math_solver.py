import numpy as np
from PIL import Image
import re
from ..utils.common import get_model_path, decode_base64_to_image, resize_with_padding, decode

def solve_math(manager, img_base64: str, math_model_path: str = '', use_gpu: bool = False):
    math_model_path = math_model_path or get_model_path('[AntiCAP]-CRNN_Math.onnx')

    session = manager.get_onnx_session(math_model_path, use_gpu)
    input_name = session.get_inputs()[0].name

    # Configuration (Must match training/export)
    IMG_H = 70
    IMG_W = 200
    CHARS = "0123456789+-*/÷×=?"
    
    image = decode_base64_to_image(img_base64).convert('RGB')
    
    # Preprocessing
    image = resize_with_padding(image, (IMG_W, IMG_H))
    
    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1)) # [C, H, W]
    
    # Normalize ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img_np = (img_np - 0.5) / 0.5
    
    # Add batch dimension
    img_np = np.expand_dims(img_np, axis=0) # [1, C, H, W]

    try:
        ort_outs = session.run(None, {input_name: img_np})
        preds = ort_outs[0] # Output 0
    except Exception as e:
        print(f"[AntiCAP] Inference error: {e}")
        return None

    # Decode
    char_map_inv = {i + 1: c for i, c in enumerate(CHARS)}
    result_str = decode(preds, char_map_inv)[0]
    
    if not result_str:
        return None

    # Standardize symbols
    expr = result_str
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
