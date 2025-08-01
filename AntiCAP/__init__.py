# coding=utf-8

import io
import os
import re
import cv2
import torch
import base64
import logging
import warnings
import ast
import numpy as np
import onnxruntime
from ultralytics import YOLO
from PIL import Image, ImageChops
import requests
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
        "charset.txt"
    ]

    os.makedirs(output_dir, exist_ok=True)

    for fname in filenames:
        filepath = os.path.join(output_dir, fname)
        if not os.path.exists(filepath):
            print(f"[Anti-CAP] Model file {fname} not found. Downloading...")
            encoded_name = urllib.parse.quote(fname)
            full_url = base_url + encoded_name
            try:
                with requests.get(full_url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"[Anti-CAP] ✅ Downloaded: {fname}")
            except Exception as e:
                print(f"[Anti-CAP] ❌ Failed to download {fname}: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise IOError(f"Failed to download required model '{fname}'.") from e


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
        detectionIcon_model_path = detectionIcon_model_path or os.path.join(os.path.dirname(__file__), 'Models', '[Icon]Detection_model.pt')
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
    def ClickIcon_Order(self, order_img_base64: str = None, target_img_base64: str = None,detectionIcon_model_path: str = '', sim_onnx_model_path: str = '',use_gpu: bool = False):

        detectionIcon_model_path = detectionIcon_model_path or os.path.join(os.path.dirname(__file__), 'Models','[Icon]Detection_model.pt')
        sim_onnx_model_path = sim_onnx_model_path or os.path.join(os.path.dirname(__file__), 'Models','[Text]Siamese_model.onnx')

        device = torch.device('cuda' if use_gpu else 'cpu')
        model = YOLO(detectionIcon_model_path, verbose=False)
        model.to(device)


        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        sim_session = onnxruntime.InferenceSession(sim_onnx_model_path, providers=providers)
        input_names = [inp.name for inp in sim_session.get_inputs()]

        def pil_to_tensor(img, size=(105, 105)):
            img = img.resize(size)
            img_np = np.asarray(img).astype(np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
            img_np = np.expand_dims(img_np, axis=0)
            return img_np

        # 解码图片
        order_image = Image.open(io.BytesIO(base64.b64decode(order_img_base64))).convert("RGB")
        target_image = Image.open(io.BytesIO(base64.b64decode(target_img_base64))).convert("RGB")

        # 图标检测
        order_results = model(order_image)
        target_results = model(target_image)

        order_boxes_list = []
        target_boxes_list = []

        if order_results and order_results[0].boxes:
            order_boxes = order_results[0].boxes.xyxy
            order_boxes_list = order_boxes.cpu().numpy().tolist()
            order_boxes_list.sort(key=lambda x: x[0])  # 从左到右排序

        if target_results and target_results[0].boxes:
            target_boxes = target_results[0].boxes.xyxy
            target_boxes_list = target_boxes.cpu().numpy().tolist()

        best_matching_boxes = []


        for order_box in order_boxes_list:
            order_crop = order_image.crop((order_box[0], order_box[1], order_box[2], order_box[3]))
            best_score = -1
            best_target_box = None

            for target_box in target_boxes_list:
                target_crop = target_image.crop((target_box[0], target_box[1], target_box[2], target_box[3]))

                tensor1 = pil_to_tensor(order_crop)
                tensor2 = pil_to_tensor(target_crop)

                inputs = {
                    input_names[0]: tensor1.astype(np.float32),
                    input_names[1]: tensor2.astype(np.float32)
                }

                output = sim_session.run(None, inputs)
                similarity_score = output[0][0][0]

                if similarity_score > best_score:
                    best_score = similarity_score
                    best_target_box = target_box

            if best_target_box:
                best_matching_boxes.append([int(coord) for coord in best_target_box])
            else:
                best_matching_boxes.append([0, 0, 0, 0])  # fallback if no match

        return best_matching_boxes


    # 文字侦测
    def Detection_Text(self, img_base64: str = None, detectionText_model_path: str = '', use_gpu: bool = False):
        detectionText_model_path = detectionText_model_path or os.path.join(os.path.dirname(__file__), 'Models', '[Text]Detection_model.pt')
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        model = YOLO(detectionText_model_path, verbose=False)
        model.to(device)


        image_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results = model(image)

        # 构建返回结果（保留两位小数）
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
    def ClickText_Order(self, order_img_base64: str = None, target_img_base64: str = None,detectionText_model_path: str = '', sim_onnx_model_path: str = '',use_gpu: bool = False):


        detectionText_model_path = detectionText_model_path or os.path.join(
            os.path.dirname(__file__), 'Models', '[Text]Detection_model.pt'
        )
        sim_onnx_model_path = sim_onnx_model_path or os.path.join(
            os.path.dirname(__file__), 'Models', '[Text]Siamese_model.onnx'
        )

        device = torch.device('cuda' if use_gpu else 'cpu')
        model = YOLO(detectionText_model_path, verbose=False)
        model.to(device)

        # 加载相似度模型（ONNX Siamese Model）
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        sim_session = onnxruntime.InferenceSession(sim_onnx_model_path, providers=providers)
        input_names = [inp.name for inp in sim_session.get_inputs()]

        def pil_to_tensor(img, size=(105, 105)):
            img = img.resize(size)
            img_np = np.asarray(img).astype(np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
            img_np = np.expand_dims(img_np, axis=0)
            return img_np

        order_image = Image.open(io.BytesIO(base64.b64decode(order_img_base64))).convert("RGB")
        target_image = Image.open(io.BytesIO(base64.b64decode(target_img_base64))).convert("RGB")

        order_results = model(order_image)
        target_results = model(target_image)

        order_boxes_list = []
        target_boxes_list = []

        if order_results and order_results[0].boxes:
            order_boxes = order_results[0].boxes.xyxy
            order_boxes_list = order_boxes.cpu().numpy().tolist()
            order_boxes_list.sort(key=lambda x: x[0])

        if target_results and target_results[0].boxes:
            target_boxes = target_results[0].boxes.xyxy
            target_boxes_list = target_boxes.cpu().numpy().tolist()

        best_matching_boxes = []

        for order_box in order_boxes_list:
            order_crop = order_image.crop((order_box[0], order_box[1], order_box[2], order_box[3]))
            best_score = -1
            best_target_box = None

            for target_box in target_boxes_list:
                target_crop = target_image.crop((target_box[0], target_box[1], target_box[2], target_box[3]))

                tensor1 = pil_to_tensor(order_crop)
                tensor2 = pil_to_tensor(target_crop)

                inputs = {
                    input_names[0]: tensor1.astype(np.float32),
                    input_names[1]: tensor2.astype(np.float32)
                }

                output = sim_session.run(None, inputs)
                similarity_score = output[0][0][0]  # 取出相似度值

                if similarity_score > best_score:
                    best_score = similarity_score
                    best_target_box = target_box

            if best_target_box:
                best_matching_boxes.append([int(coord) for coord in best_target_box])
            else:
                best_matching_boxes.append([0, 0, 0, 0])  # fallback

        return best_matching_boxes


    # 缺口滑块
    def Slider_Match(self, target_base64: str = None, background_base64: str = None, simple_target: bool = False, flag: bool = False):

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

                # Ensure start and end points are valid
                if starttx >= end_x or startty >= end_y:
                    return None, 0, 0  # Or raise an exception

                return image.crop([starttx, startty, end_x, end_y]), starttx, startty
            except Exception as e:
                # print(f"Error in get_target: {e}")
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
    def compare_image_similarity(self,image1_base64: str = None,image2_base64: str = None,sim_onnx_model_path: str = '',use_gpu: bool = False):

        sim_onnx_model_path = sim_onnx_model_path or os.path.join(os.path.dirname(__file__), 'Models', '[Text]Siamese_model.onnx')

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

