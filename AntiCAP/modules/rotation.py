
import math
import numpy as np
import cv2
from PIL import Image

from ..utils.common import get_model_path, decode_base64_to_image, decode_base64_to_cv2

def solve_single_rotate(manager, img_base64: str, rotate_onnx_modex_path: str = '', use_gpu: bool = False):
    rotate_onnx_modex_path = rotate_onnx_modex_path or get_model_path('[AntiCAP]-Rotation-RotNetR.onnx')

    session = manager.get_onnx_session(rotate_onnx_modex_path, use_gpu)

    img = decode_base64_to_image(img_base64).convert("RGB")

    DEFAULT_TARGET_SIZE = 224
    SQRT2 = math.sqrt(2.0)

    # PIL -> numpy, CHW
    img_np = np.array(img, dtype=np.uint8)
    img_np = np.transpose(img_np, (2, 0, 1))

    _, h, w = img_np.shape
    assert h == w, "Image must be square"
    new_size = int(h / SQRT2)
    top = (h - new_size) // 2
    left = (w - new_size) // 2
    img_np = img_np[:, top:top + new_size, left:left + new_size]

    img_np = img_np.astype(np.float32) / 255.0

    img_tmp = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
    img_tmp = Image.fromarray((img_tmp * 255).astype(np.uint8))
    img_tmp = img_tmp.resize((DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE), Image.Resampling.BILINEAR)
    img_np = np.array(img_tmp, dtype=np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW

    # Normalize (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img_np = (img_np - mean) / std

    img_np = np.expand_dims(img_np, axis=0)

    ort_inputs = {session.get_inputs()[0].name: img_np}
    predict = session.run(None, ort_inputs)[0]

    degree = int(np.argmax(predict, axis=1).item())
    return degree

def solve_double_rotate(manager, inside_base64: str, outside_base64: str, check_pixel: int = 10, speed_ratio: float = 1, grayscale: bool = False, anticlockwise: bool = False, cut_pixel_value: int = 0):
    image_array_inner = decode_base64_to_cv2(inside_base64)
    if image_array_inner is None:
        raise ValueError("[AntiCAP] Failed to decode inside_base64 image")
    
    if len(image_array_inner.shape) == 2:
        inner_image = cv2.cvtColor(image_array_inner, cv2.COLOR_GRAY2BGR)
    elif image_array_inner.shape[2] == 4:
        inner_image = cv2.cvtColor(image_array_inner, cv2.COLOR_BGRA2BGR)
    else:
        inner_image = image_array_inner

    if grayscale:
        inner_image = cv2.cvtColor(inner_image, cv2.COLOR_BGR2GRAY)

    image_array_outer = decode_base64_to_cv2(outside_base64)
    if image_array_outer is None:
        raise ValueError("[AntiCAP] Failed to decode outside_base64 image")

    if len(image_array_outer.shape) == 2:
        outer_image = cv2.cvtColor(image_array_outer, cv2.COLOR_GRAY2BGR)
    elif image_array_outer.shape[2] == 4:
        outer_image = cv2.cvtColor(image_array_outer, cv2.COLOR_BGRA2BGR)
    else:
        outer_image = image_array_outer

    if grayscale:
        outer_image = cv2.cvtColor(outer_image, cv2.COLOR_BGR2GRAY)

    cut_pixel_list_inner = []
    height_inner, width_inner = inner_image.shape[:2]
    for rotate_count in range(4):
        cut_pixel = 0
        rotate_array = np.rot90(inner_image, rotate_count).copy()
        for line in rotate_array:
            if len(line.shape) == 1:  # grayscale
                pixel_set = set(line.tolist()) - {0, 255}
            else:  # color
                pixel_set = set(map(tuple, line)) - {(0, 0, 0), (255, 255, 255)}
            if not pixel_set:
                cut_pixel += 1
            else:
                break
        cut_pixel_list_inner.append(cut_pixel)

    cut_pixel_list_inner[2] = height_inner - cut_pixel_list_inner[2]
    cut_pixel_list_inner[3] = width_inner - cut_pixel_list_inner[3]
    up_inner, left_inner, down_inner, right_inner = cut_pixel_list_inner

    cut_array_inner = inner_image[up_inner:down_inner, left_inner:right_inner]
    if cut_array_inner.size == 0:
        raise ValueError("[AntiCAP] cut_array_inner 是空的，请检查输入图片或裁剪逻辑。")

    diameter_inner = (min(cut_array_inner.shape[:2]) // 2) * 2
    cut_inner_image = cv2.resize(cut_array_inner, dsize=(diameter_inner, diameter_inner))
    cut_inner_radius = cut_inner_image.shape[0] // 2

    cut_pixel_list_outer = []
    height_outer, width_outer = outer_image.shape[:2]
    y, x = height_outer // 2, width_outer // 2
    resize_check_pixel = int(math.ceil(cut_inner_radius / (cut_inner_radius - check_pixel) * check_pixel))
    for i in (-1, 1):
        for p in (y, x):
            pos = p + i * cut_inner_radius
            for _ in range(p - cut_inner_radius):
                p_x, p_y = (pos, y) if len(cut_pixel_list_outer) % 2 else (x, pos)
                pixel_point = outer_image[p_y][p_x]
                if isinstance(pixel_point, np.uint8):
                    pixel_set = {int(pixel_point)} - {0, 255}
                else:
                    pixel_set = {tuple(pixel_point)} - {(0, 0, 0), (255, 255, 255)}
                if not pixel_set:
                    pos += i
                    continue
                status = True
                for pixel in pixel_set:
                    if isinstance(pixel, int):
                        if pixel <= cut_pixel_value or pixel >= 255 - cut_pixel_value:
                            status = False
                            break
                    else:  # tuple RGB
                        if any(v <= cut_pixel_value or v >= 255 - cut_pixel_value for v in pixel):
                            status = False
                            break
                if status:
                    break
                pos += i
            cut_pixel_list_outer.append(pos + i * resize_check_pixel)

    up_outer, left_outer, down_outer, right_outer = cut_pixel_list_outer

    cut_array_outer = outer_image[up_outer:down_outer, left_outer:right_outer]
    if cut_array_outer.size == 0:
        raise ValueError("[AntiCAP] cut_array_outer 是空的，请检查输入图片或裁剪逻辑。")

    diameter_outer = (min(cut_array_outer.shape[:2]) // 2) * 2
    cut_outer_image = cv2.resize(cut_array_outer, dsize=(diameter_outer, diameter_outer))

    radius_inner = cut_inner_image.shape[0] // 2
    center_point_inner = (radius_inner, radius_inner)
    mask_inner = np.zeros((radius_inner * 2, radius_inner * 2), dtype=np.uint8)
    cv2.circle(mask_inner, center_point_inner, radius_inner, 255, -1)
    cv2.circle(mask_inner, center_point_inner, radius_inner - check_pixel, 0, -1)
    src_array_inner = np.zeros_like(cut_inner_image)
    inner_annulus = cv2.add(cut_inner_image, src_array_inner, mask=mask_inner)

    radius_outer = cut_outer_image.shape[0] // 2
    center_point_outer = (radius_outer, radius_outer)
    mask_outer = np.zeros((radius_outer * 2, radius_outer * 2), dtype=np.uint8)
    cv2.circle(mask_outer, center_point_outer, radius_outer, 255, -1)
    cv2.circle(mask_outer, center_point_outer, radius_outer - check_pixel, 0, -1)
    src_array_outer = np.zeros_like(cut_outer_image)
    outer_annulus = cv2.add(cut_outer_image, src_array_outer, mask=mask_outer)

    rotate_info_list = [{'similar': 0, 'angle': 0, 'start': 1, 'end': 361, 'step': 10}]
    rtype = -1 if anticlockwise else 1
    h, w = inner_annulus.shape[:2]

    for item in rotate_info_list:
        for angle in range(item['start'], item['end'], item['step']):
            mat_rotate = cv2.getRotationMatrix2D((h * 0.5, w * 0.5), rtype * angle, 1)
            dst = cv2.warpAffine(inner_annulus, mat_rotate, (h, w))
            ret = cv2.matchTemplate(outer_annulus, dst, cv2.TM_CCOEFF_NORMED)
            similar_value = cv2.minMaxLoc(ret)[1]
            if similar_value < min(rotate_info_list, key=lambda x: x['similar'])['similar']:
                continue
            rotate_info = {
                'similar': similar_value,
                'angle': angle,
                'start': angle - 10,
                'end': angle + 10,
                'step': 10
            }
            rotate_info_list.append(rotate_info)
            if len(rotate_info_list) > 5:
                min_index = min(range(len(rotate_info_list)), key=lambda i: rotate_info_list[i]['similar'])
                rotate_info_list.pop(min_index)

    best_rotate_info = max(rotate_info_list, key=lambda x: x['similar'])

    inner_angle = round(best_rotate_info['angle'] * speed_ratio / (speed_ratio + 1), 2)
    return {
        "similarity": best_rotate_info['similar'],
        "inner_angle": inner_angle,
        "raw_angle": best_rotate_info['angle']
    }
