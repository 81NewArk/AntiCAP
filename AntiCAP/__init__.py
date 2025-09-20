# coding=utf-8

import io
import os
import re
import cv2
import ast
import math
import torch
import base64
import logging
import warnings
import requests
import onnxruntime
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image, ImageChops

import urllib.parse

warnings.filterwarnings('ignore')
onnxruntime.set_default_logger_severity(3)


def Download_Models_if_needed():
    current_dir = os.path.dirname(__file__)
    output_dir = os.path.join(current_dir, "Models")

    base_url = "https://hf-mirror.com/spaces/NewArk81/AntiCAP_Models/resolve/main/"
    filenames = [
        "[Icon]Detection_model.pt",
        "[Math]Detection_model.pt",
        "[OCR]Ddddocr.onnx",
        "[Text]Detection_model.pt",
        "[Text]Siamese_model.onnx",
        "[Rotate]Rotate_model.onnx",
        "charset.txt"
    ]

    os.makedirs(output_dir, exist_ok=True)

    print(f"[Anti-CAP] 首次使用，正在检查模型文件...")
    for fname in filenames:
        filepath = os.path.join(output_dir, fname)
        if not os.path.exists(filepath):
            print(f"[Anti-CAP] ⚠️ 模型文件 '{fname}' 不存在，正在下载...")
            encoded_name = urllib.parse.quote(fname)
            full_url = base_url + encoded_name
            try:
                with requests.get(full_url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get("Content-Length", 0))
                    with open(filepath, "wb") as f, tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=fname,
                        ncols=80,
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                bar.update(len(chunk))
                print(f"[Anti-CAP] ✅ 模型文件 '{fname}' 下载完成。")
            except Exception as e:
                print(f"[Anti-CAP] ❌ 模型文件 '{fname}' 下载失败: {e}")
                print(f"[Anti-CAP] 请手动下载模型文件 '{fname}' 并放置在目录 '{output_dir}' 中，或检查网络连接后重试。")
                print(f"[Anti-CAP] 下载链接: https://github.com/81NewArk/AntiCAP/tree/main/AntiCAP/Models")
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise IOError(f"无法下载模型文件 '{fname}'，请检查网络连接或稍后重试。")

class TypeError(Exception):
    pass


class Handler(object):
    logging.getLogger('ultralytics').setLevel(logging.WARNING)

    def __init__(self, show_banner=True):
        Download_Models_if_needed()

        if show_banner:
            print('''
            -----------------------------------------------------------  
            |      _              _     _    ____      _      ____    |
            |     / \     _ __   | |_  (_)  / ___|    / \    |  _ \   |
            |    / _ \   | '_ \  | __| | | | |       / _ \   | |_) |  |
            |   / ___ \  | | | | | |_  | | | |___   / ___ \  |  __/   |
            |  /_/   \_\ |_| |_|  \__| |_|  \____| /_/   \_\ |_|      |
            ----------------------------------------------------------- 
            |         Github: https://github.com/81NewArk/AntiCAP     |
            |         Author: 81NewArk                                |
            -----------------------------------------------------------''')

    # 文字识别
    def OCR(self, img_base64: str = None, use_gpu: bool = False, png_fix: bool = False, probability=False):

        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'Models', '[OCR]Ddddocr.onnx')
        charset_path = os.path.join(current_dir, 'Models', 'charset.txt')

        try:
            with open(charset_path, 'r', encoding='utf-8') as f:
                list_as_string = f.read()
                charset = ast.literal_eval(list_as_string)
        except FileNotFoundError:
            raise FileNotFoundError(f"字符集文件未在 {charset_path} 找到。")
        except Exception as e:
            raise ValueError(f"解析字符集文件时出错: {e}")

        providers = ['CUDAExecutionProvider'] if use_gpu and onnxruntime.get_device().upper() == 'GPU' else [
            'CPUExecutionProvider']
        session = onnxruntime.InferenceSession(model_path, providers=providers)

        img_data = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_data))

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
            # 应用 softmax 进行概率计算
            ort_outs = np.exp(ort_outs) / np.sum(np.exp(ort_outs), axis=2, keepdims=True)
            ort_outs_probability = np.squeeze(ort_outs).tolist()

            result = {
                'charsets': charset,
                'probability': ort_outs_probability
            }
            return result

    # 算术识别
    def Math(self, img_base64: str, math_model_path: str = '', use_gpu: bool = False):

        math_model_path = math_model_path or os.path.join(os.path.dirname(__file__), 'Models','[Math]Detection_model.pt')

        device = torch.device('cuda' if use_gpu else 'cpu')
        model = YOLO(math_model_path, verbose=False)
        model.to(device)

        image_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(image_bytes))
        results = model(image)

        sorted_elements = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            x1 = float(box.xyxy[0][0])
            sorted_elements.append((x1, label))

        sorted_elements.sort(key=lambda x: x[0])
        sorted_labels = [label for _, label in sorted_elements]

        captcha_text = ''.join(sorted_labels)
        result = None

        if captcha_text:
            expr = captcha_text.split('=')[0] if '=' in captcha_text else captcha_text
            expr = expr.replace('×', '*').replace('÷', '/')
            expr = re.sub(r'[^\d\+\-\*/]', '', expr)
            try:
                result = eval(expr)
            except Exception as e:
                print(f"[Anti-CAP] 表达式解析出错: {e}")
        else:
            print("[Anti-CAP] 识别失败，未获取到表达式")

        return result

    # 图标侦测
    def Detection_Icon(self, img_base64: str = None, detectionIcon_model_path: str = '', use_gpu: bool = False):
        detectionIcon_model_path = detectionIcon_model_path or os.path.join(os.path.dirname(__file__), 'Models','[Icon]Detection_model.pt')
        device = torch.device('cuda' if use_gpu else 'cpu')
        model = YOLO(detectionIcon_model_path, verbose=False)
        model.to(device)

        image_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(image_bytes))

        results = model(image)

        detections = []
        for box in results[0].boxes:
            coords = box.xyxy[0].tolist()
            rounded_box = [round(coord, 2) for coord in coords]
            class_name = results[0].names[int(box.cls[0])]
            detections.append({
                'class': class_name,
                'box': rounded_box
            })

        return detections

    # 按序侦测图标
    def ClickIcon_Order(self, order_img_base64: str, target_img_base64: str, detectionIcon_model_path: str = '',sim_onnx_model_path: str = '', use_gpu: bool = False):
        detectionIcon_model_path = detectionIcon_model_path or os.path.join(os.path.dirname(__file__), 'Models','[Icon]Detection_model.pt')
        sim_onnx_model_path = sim_onnx_model_path or os.path.join(os.path.dirname(__file__), 'Models','[Text]Siamese_model.onnx')

        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        model = YOLO(detectionIcon_model_path)
        model.to(device)

        providers = ['CUDAExecutionProvider'] if use_gpu and onnxruntime.get_device().upper() == 'GPU' else [
            'CPUExecutionProvider']
        sim_session = onnxruntime.InferenceSession(sim_onnx_model_path, providers=providers)
        input_names = [inp.name for inp in sim_session.get_inputs()]

        def pil_to_tensor(img, size=(105, 105)):
            img = img.convert("RGB").resize(size)
            img_np = np.asarray(img).astype(np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
            return np.expand_dims(img_np, axis=0)

        order_image = Image.open(io.BytesIO(base64.b64decode(order_img_base64))).convert("RGB")
        target_image = Image.open(io.BytesIO(base64.b64decode(target_img_base64))).convert("RGB")

        order_results = model(order_image, verbose=False)
        target_results = model(target_image, verbose=False)

        order_boxes_list = []
        if order_results and order_results[0].boxes:
            order_boxes = order_results[0].boxes.xyxy.cpu().numpy().tolist()
            order_boxes.sort(key=lambda x: x[0])  # 按X坐标从左到右
            order_boxes_list = order_boxes

        target_boxes_list = []
        if target_results and target_results[0].boxes:
            target_boxes_list = target_results[0].boxes.xyxy.cpu().numpy().tolist()

        available_target_boxes = target_boxes_list.copy()
        best_matching_boxes = []

        for order_box in order_boxes_list:
            order_crop = order_image.crop(order_box)
            order_tensor = pil_to_tensor(order_crop)

            best_score = -1
            best_target_box = None

            for target_box in available_target_boxes:
                target_crop = target_image.crop(target_box)
                target_tensor = pil_to_tensor(target_crop)

                inputs = {input_names[0]: order_tensor, input_names[1]: target_tensor}
                output = sim_session.run(None, inputs)
                similarity_score = output[0][0][0]

                if similarity_score > best_score:
                    best_score = similarity_score
                    best_target_box = target_box

            if best_target_box:
                best_matching_boxes.append([int(coord) for coord in best_target_box])
                available_target_boxes.remove(best_target_box)
            else:
                best_matching_boxes.append([0, 0, 0, 0])

        return best_matching_boxes

    # 文字侦测
    def Detection_Text(self, img_base64: str = None, detectionText_model_path: str = '', use_gpu: bool = False):

        detectionText_model_path = detectionText_model_path or os.path.join(os.path.dirname(__file__), 'Models','[Text]Detection_model.pt')
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        model = YOLO(detectionText_model_path, verbose=False)
        model.to(device)

        image_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results = model(image)

        detections = []
        for box in results[0].boxes:
            coords = box.xyxy[0].tolist()
            rounded_box = [round(coord, 2) for coord in coords]
            class_name = results[0].names[int(box.cls[0])]
            detections.append({
                'class': class_name,
                'box': rounded_box
            })

        return detections

    # 按序侦测文字
    def ClickText_Order(self, order_img_base64: str, target_img_base64: str, detectionText_model_path: str = '',sim_onnx_model_path: str = '', use_gpu: bool = False):

        detectionText_model_path = detectionText_model_path or os.path.join(os.path.dirname(__file__), 'Models','[Text]Detection_model.pt')

        sim_onnx_model_path = sim_onnx_model_path or os.path.join(os.path.dirname(__file__), 'Models','[Text]Siamese_model.onnx')

        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        model = YOLO(detectionText_model_path)
        model.to(device)

        providers = ['CUDAExecutionProvider'] if use_gpu and onnxruntime.get_device().upper() == 'GPU' else [
            'CPUExecutionProvider']
        sim_session = onnxruntime.InferenceSession(sim_onnx_model_path, providers=providers)
        input_names = [inp.name for inp in sim_session.get_inputs()]

        def pil_to_tensor(img, size=(105, 105)):
            img = img.convert("RGB").resize(size)
            img_np = np.asarray(img).astype(np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
            return np.expand_dims(img_np, axis=0)

        order_image = Image.open(io.BytesIO(base64.b64decode(order_img_base64))).convert("RGB")
        target_image = Image.open(io.BytesIO(base64.b64decode(target_img_base64))).convert("RGB")

        order_results = model(order_image, verbose=False)
        target_results = model(target_image, verbose=False)

        order_boxes_list = []
        if order_results and order_results[0].boxes:
            order_boxes = order_results[0].boxes.xyxy.cpu().numpy().tolist()
            order_boxes.sort(key=lambda x: x[0])  # 左到右排序
            order_boxes_list = order_boxes

        target_boxes_list = []
        if target_results and target_results[0].boxes:
            target_boxes_list = target_results[0].boxes.xyxy.cpu().numpy().tolist()

        available_target_boxes = target_boxes_list.copy()
        best_matching_boxes = []

        for order_box in order_boxes_list:
            order_crop = order_image.crop(order_box)
            order_tensor = pil_to_tensor(order_crop)

            best_score = -1
            best_target_box = None

            for target_box in available_target_boxes:
                target_crop = target_image.crop(target_box)
                target_tensor = pil_to_tensor(target_crop)

                inputs = {input_names[0]: order_tensor, input_names[1]: target_tensor}
                output = sim_session.run(None, inputs)
                similarity_score = output[0][0][0]

                if similarity_score > best_score:
                    best_score = similarity_score
                    best_target_box = target_box

            if best_target_box:
                best_matching_boxes.append([int(coord) for coord in best_target_box])
                available_target_boxes.remove(best_target_box)
            else:
                best_matching_boxes.append([0, 0, 0, 0])

        return best_matching_boxes

    # 缺口滑块
    def Slider_Match(self, target_base64: str = None, background_base64: str = None, simple_target: bool = False,flag: bool = False):

        def get_target(img_bytes: bytes = None):
            try:
                image = Image.open(io.BytesIO(img_bytes))
                w, h = image.size
                starttx = 0
                startty = 0
                end_x = 0
                end_y = 0
                found_alpha = False
                for y in range(h):
                    row_has_alpha = False
                    for x in range(w):
                        p = image.getpixel((x, y))
                        if len(p) == 4 and p[-1] < 255:
                            row_has_alpha = True
                            found_alpha = True
                            if startty == 0:
                                startty = y
                            break
                    if found_alpha and not row_has_alpha and end_y == 0 and startty != 0:
                        end_y = y
                        break
                    elif found_alpha and y == h - 1 and end_y == 0:
                        end_y = h

                found_alpha_in_row = False
                for x in range(w):
                    col_has_alpha = False
                    for y in range(h):
                        p = image.getpixel((x, y))
                        if len(p) == 4 and p[-1] < 255:
                            col_has_alpha = True
                            found_alpha_in_row = True
                            if starttx == 0:
                                starttx = x
                            break
                    if found_alpha_in_row and not col_has_alpha and end_x == 0 and starttx != 0:
                        end_x = x
                        break
                    elif found_alpha_in_row and x == w - 1 and end_x == 0:
                        end_x = w

                if end_x == 0 and starttx != 0:
                    end_x = w
                if end_y == 0 and startty != 0:
                    end_y = h

                if starttx >= end_x or startty >= end_y:
                    return None, 0, 0

                return image.crop([starttx, startty, end_x, end_y]), starttx, startty
            except Exception as e:
                return None, 0, 0

        def decode_base64_to_image(base64_string):
            try:
                image_data = base64.b64decode(base64_string)
                img_array = np.frombuffer(image_data, np.uint8)
                return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            except Exception as e:
                print(f"Error decoding base64: {e}")
                return None

        if not simple_target:
            target_image = decode_base64_to_image(target_base64)
            if target_image is None:
                if flag:
                    raise ValueError("Failed to decode target base64 image.")
                return self.Slider_Match(target_base64=target_base64,
                                         background_base64=background_base64,
                                         simple_target=True, flag=True)
            try:
                target_pil, target_x, target_y = get_target(target_image.tobytes())
                if target_pil is None:
                    if flag:
                        raise ValueError("Failed to extract target from image.")
                    return self.Slider_Match(target_base64=target_base64,
                                             background_base64=background_base64,
                                             simple_target=True, flag=True)
                target = cv2.cvtColor(np.asarray(target_pil), cv2.COLOR_RGB2BGR)
            except SystemError as e:
                if flag:
                    raise e
                return self.Slider_Match(target_base64=target_base64,
                                         background_base64=background_base64,
                                         simple_target=True, flag=True)
        else:
            target = decode_base64_to_image(target_base64)
            if target is None:
                return {"target_x": 0, "target_y": 0, "target": [0, 0, 0, 0]}
            target_y = 0
            target_x = 0

        background = decode_base64_to_image(background_base64)
        if background is None:
            return {"target_x": target_x, "target_y": target_y, "target": [0, 0, 0, 0]}

        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        background_canny = cv2.Canny(background_gray, 100, 200)
        target_canny = cv2.Canny(target_gray, 100, 200)

        background_rgb = cv2.cvtColor(background_canny, cv2.COLOR_GRAY2BGR)
        target_rgb = cv2.cvtColor(target_canny, cv2.COLOR_GRAY2BGR)

        res = cv2.matchTemplate(background_rgb, target_rgb, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        h, w = target_rgb.shape[:2]
        bottom_right = (max_loc[0] + w, max_loc[1] + h)

        return {"target_x": target_x,
                "target_y": target_y,
                "target": [int(max_loc[0]), int(max_loc[1]), int(bottom_right[0]), int(bottom_right[1])]}

    # 阴影滑块
    def Slider_Comparison(self, target_base64: str = None, background_base64: str = None):
        def decode_base64_to_image(base64_string):
            image_data = base64.b64decode(base64_string)
            return Image.open(io.BytesIO(image_data)).convert("RGB")

        target = decode_base64_to_image(target_base64)
        background = decode_base64_to_image(background_base64)

        image = ImageChops.difference(background, target)
        background.close()
        target.close()
        image = image.point(lambda x: 255 if x > 80 else 0)
        start_y = 0
        start_x = 0

        for i in range(0, image.width):
            count = 0
            for j in range(0, image.height):
                pixel = image.getpixel((i, j))
                if pixel != (0, 0, 0):
                    count += 1
                if count >= 5 and start_y == 0:
                    start_y = j - 5

            if count >= 5:
                start_x = i + 2
                break

        return {
            "target": [start_x, start_y]
        }

    # 图像相似度比较 对比图片的中的文字
    def Compare_Image_Similarity(self, image1_base64: str = None, image2_base64: str = None,sim_onnx_model_path: str = '', use_gpu: bool = False):

        sim_onnx_model_path = sim_onnx_model_path or os.path.join(os.path.dirname(__file__), 'Models','[Text]Siamese_model.onnx')

        def decode_base64_to_pil(b64str):
            img_bytes = base64.b64decode(b64str)
            return Image.open(io.BytesIO(img_bytes)).convert('RGB')

        def pil_to_numpy(img, size=(105, 105)):
            img = img.resize(size)
            img_np = np.asarray(img).astype(np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))
            img_np = np.expand_dims(img_np, axis=0)
            return img_np

        img1 = decode_base64_to_pil(image1_base64)
        img2 = decode_base64_to_pil(image2_base64)

        tensor1 = pil_to_numpy(img1)
        tensor2 = pil_to_numpy(img2)

        # 设置推理设备
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

        ort_session = onnxruntime.InferenceSession(sim_onnx_model_path, providers=providers)
        input_names = [inp.name for inp in ort_session.get_inputs()]

        inputs = {
            input_names[0]: tensor1.astype(np.float32),
            input_names[1]: tensor2.astype(np.float32)
        }

        outputs = ort_session.run(None, inputs)
        similarity = outputs[0][0][0]

        return similarity

    # 单图旋转角度
    def Single_Rotate(self, img_base64: str, rotate_onnx_modex_path: str = '',use_gpu: bool = False):

        rotate_onnx_modex_path = rotate_onnx_modex_path or os.path.join(os.path.dirname(__file__), 'Models', '[Rotate]Rotate_model.onnx')

        providers = ['CUDAExecutionProvider'] if use_gpu and onnxruntime.get_device().upper() == 'GPU' else ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(rotate_onnx_modex_path, providers=providers)


        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")


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

        # 归一化 (ImageNet mean/std)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_np = (img_np - mean) / std


        img_np = np.expand_dims(img_np, axis=0)


        ort_inputs = {session.get_inputs()[0].name: img_np}
        predict = session.run(None, ort_inputs)[0]

        degree = int(np.argmax(predict, axis=1).item())
        return degree

    # 双图旋转
    def Double_Rotate(self,inside_base64: str, outside_base64: str, check_pixel: int = 10, speed_ratio: float = 1,grayscale: bool = False, anticlockwise: bool = False, cut_pixel_value: int = 0, ):
        image_array_inner = np.asarray(bytearray(base64.b64decode(inside_base64)), dtype="uint8")
        inner_image = cv2.imdecode(image_array_inner, 1)
        if grayscale:
            inner_image = cv2.cvtColor(inner_image, cv2.COLOR_BGR2GRAY)

        image_array_outer = np.asarray(bytearray(base64.b64decode(outside_base64)), dtype="uint8")
        outer_image = cv2.imdecode(image_array_outer, 1)
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
                    break  # 遇到非空像素就停止
            cut_pixel_list_inner.append(cut_pixel)

        cut_pixel_list_inner[2] = height_inner - cut_pixel_list_inner[2]
        cut_pixel_list_inner[3] = width_inner - cut_pixel_list_inner[3]
        up_inner, left_inner, down_inner, right_inner = cut_pixel_list_inner

        cut_array_inner = inner_image[up_inner:down_inner, left_inner:right_inner]
        if cut_array_inner.size == 0:
            raise ValueError("[Anti-CAP] cut_array_inner 是空的，请检查输入图片或裁剪逻辑。")

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
            raise ValueError("[Anti-CAP] cut_array_outer 是空的，请检查输入图片或裁剪逻辑。")

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
