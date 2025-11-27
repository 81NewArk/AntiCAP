import numpy as np
from PIL import Image

from ..utils.common import get_model_path, decode_base64_to_image

def solve_ocr(manager, img_base64: str, use_gpu: bool = False, png_fix: bool = False, probability=False):
    model_path = get_model_path('[Dddd]-OCR.onnx')
    charset_path = get_model_path('[Dddd]-CharSets.txt')

    charset = manager.get_charset(charset_path)
    session = manager.get_onnx_session(model_path, use_gpu)

    image = decode_base64_to_image(img_base64)

    image = image.resize((int(image.size[0] * (64 / image.size[1])), 64), Image.Resampling.LANCZOS).convert('L')
    image = np.array(image).astype(np.float32)
    image = np.expand_dims(image, axis=0) / 255.
    image = (image - 0.5) / 0.5

    ort_inputs = {'input1': np.array([image]).astype(np.float32)}
    ort_outs = session.run(None, ort_inputs)

    result = []
    last_item = 0

    if not probability:
        argmax_result = np.squeeze(np.argmax(ort_outs[0], axis=2))
        for item in argmax_result:
            if item == last_item:
                continue
            else:
                last_item = item
            if item != 0:
                result.append(charset[item])

        return ''.join(result)
    else:
        ort_outs = ort_outs[0]
        ort_outs = np.exp(ort_outs) / np.sum(np.exp(ort_outs), axis=2, keepdims=True)
        ort_outs_probability = np.squeeze(ort_outs).tolist()

        result = {
            'charsets': charset,
            'probability': ort_outs_probability
        }
        return result
