import json
import numpy as np
from PIL import Image

MODEL_META_CACHE = {}

def get_model_meta(model_path, session):
    if model_path in MODEL_META_CACHE:
        return MODEL_META_CACHE[model_path]
    
    model_meta = session.get_modelmeta()
    mean = [0.485, 0.456, 0.406] # Default
    std = [0.229, 0.224, 0.225] # Default
    
    mean_np = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std_np = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    if 'mean' in model_meta.custom_metadata_map and 'std' in model_meta.custom_metadata_map:
        try:
            mean_np = np.array(json.loads(model_meta.custom_metadata_map['mean']), dtype=np.float32).reshape(3, 1, 1)
            std_np = np.array(json.loads(model_meta.custom_metadata_map['std']), dtype=np.float32).reshape(3, 1, 1)
        except Exception:
            print("[AntiCAP] 提示：解析自定义模型的 mean/std 失败，使用默认值。")

    input_meta = session.get_inputs()[0]
    try:
        input_size = (input_meta.shape[3], input_meta.shape[2])
    except (IndexError, TypeError):
        print("[AntiCAP] 提示：无法从模型元数据推断输入尺寸，使用默认值 (224, 224)。")
        input_size = (224, 224)

    meta = {'mean': mean_np, 'std': std_np, 'input_size': input_size}
    MODEL_META_CACHE[model_path] = meta
    return meta

def get_siamese_similarity(manager, image1: Image.Image, image2: Image.Image, model_path: str, use_gpu: bool):
    session = manager.get_onnx_session(model_path, use_gpu)
    
    # Get meta locally
    meta = get_model_meta(model_path, session)

    def preprocess(img):
        img = img.convert('RGB').resize(meta['input_size'], Image.Resampling.LANCZOS)
        tensor = np.array(img, dtype=np.float32) / 255.0
        tensor = (tensor.transpose(2, 0, 1) - meta['mean']) / meta['std']
        return np.expand_dims(tensor, axis=0)

    tensor1, tensor2 = preprocess(image1), preprocess(image2)
    input_feed = {
        session.get_inputs()[0].name: tensor1,
        session.get_inputs()[1].name: tensor2
    }

    outputs = session.run(None, input_feed)
    emb1, emb2 = outputs[0], outputs[1]
    dist = np.linalg.norm(emb1 - emb2)
    similarity = 1 / (1 + dist)
    return similarity

from ..utils.common import get_model_path, decode_base64_to_image

def solve_compare_image_similarity(manager, image1_base64: str, image2_base64: str, model_path: str = None, use_gpu: bool = False):
    model_path = model_path or get_model_path('[AntiCAP]-Siamese-ResNet18.onnx')

    image1 = decode_base64_to_image(image1_base64)
    image2 = decode_base64_to_image(image2_base64)

    return get_siamese_similarity(manager, image1, image2, model_path, use_gpu)
