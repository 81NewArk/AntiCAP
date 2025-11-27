import json
import numpy as np
from PIL import Image
from ..utils.common import get_model_path, decode_base64_to_image


MODEL_META_CACHE = {}

def get_model_meta(model_path, session, is_custom_model=False):
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
    elif is_custom_model:
        print('''[AntiCAP] 提示：为了兼容本项目，您的自定义 ONNX 模型必须包含 `mean` 和 `std` 元数据。
                
                ⚠️ 如果您的训练归一化参数与默认值不同，请务必在导出 ONNX 时添加正确的 `mean` 和 `std`，否则图像相似度计算可能不准确。
                1. 在 ONNX 模型中添加自定义元数据 `mean` 和 `std`，值为列表形式，例如 [0.485,0.456,0.406] 和 [0.229,0.224,0.225]。
                2. 保存模型后，项目会自动读取这些元数据进行归一化处理，从而保证相似度计算精度与兼容性。''')
    
    input_meta = session.get_inputs()[0]
    
    if len(input_meta.shape) == 4 and isinstance(input_meta.shape[2], int) and isinstance(input_meta.shape[3], int):
        input_size = (input_meta.shape[3], input_meta.shape[2])
    else:
        input_size = (105, 105) # Default for Siamese-ResNet18

    meta = {'mean': mean_np, 'std': std_np, 'input_size': input_size}
    MODEL_META_CACHE[model_path] = meta
    return meta

def get_siamese_similarity(manager, image1: Image.Image, image2: Image.Image, model_path: str, use_gpu: bool, is_custom_model: bool = False):
    session = manager.get_onnx_session(model_path, use_gpu)
    
    # Get meta locally
    meta = get_model_meta(model_path, session, is_custom_model)

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



def solve_compare_image_similarity(manager, image1_base64: str, image2_base64: str, sim_onnx_model_path: str = None, use_gpu: bool = False):
    
    if sim_onnx_model_path:
        model_path = sim_onnx_model_path
    else:
        model_path = get_model_path('[AntiCAP]-Siamese-ResNet18.onnx')

    image1 = decode_base64_to_image(image1_base64)
    image2 = decode_base64_to_image(image2_base64)

    return get_siamese_similarity(manager, image1, image2, model_path, use_gpu, is_custom_model=bool(sim_onnx_model_path))
