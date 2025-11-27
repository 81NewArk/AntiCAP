# coding=utf-8
import os
import time
import urllib.parse
import requests
import onnxruntime
import torch
import ast
from tqdm import tqdm
from ultralytics import YOLO

class ModelManager:
    def __init__(self):
        self._download_models_if_needed()
        self.models = {}
        self.charsets = {}
        self.charsets = {}

        onnxruntime.set_default_logger_severity(3)

    def _download_models_if_needed(self):
        current_dir = os.path.dirname(__file__)
        # Go up one level to AntiCAP root
        package_root = os.path.dirname(current_dir)
        output_dir = os.path.join(package_root, "AntiCAP-Models")

        base_url = "https://newark81.vip/AntiCAP-Models/"
        filenames = [
            "[AntiCAP]-Detection_Icon-YOLO.pt",
            "[AntiCAP]-CRNN_Math.onnx",
            "[AntiCAP]-Detection_Text-YOLO.pt",
            "[AntiCAP]-Siamese-ResNet18.onnx",
            "[AntiCAP]-Rotation-RotNetR.onnx",
            "[Dddd]-OCR.onnx",
            "[Dddd]-CharSets.txt",
        ]

        os.makedirs(output_dir, exist_ok=True)

        MAX_RETRIES = 3

        for fname in filenames:
            filepath = os.path.join(output_dir, fname)
            if os.path.exists(filepath):
                continue

            print(f"[AntiCAP] ⚠️ 模型文件 '{fname}' 不存在，正在下载...")

            encoded_name = urllib.parse.quote(fname)
            full_url = base_url + encoded_name

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resume_header = {}
                    mode = "wb"
                    existing_size = 0
                    if os.path.exists(filepath):
                        existing_size = os.path.getsize(filepath)
                        if existing_size > 0:
                            resume_header = {"Range": f"bytes={existing_size}-"}
                            mode = "ab"

                    with requests.get(full_url, headers=resume_header, stream=True, timeout=300) as r:
                        r.raise_for_status()
                        total_size = int(r.headers.get("Content-Length", 0))
                        total_size += existing_size

                        with open(filepath, mode) as f, tqdm(
                            total=total_size,
                            initial=existing_size,
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

                    print(f"[AntiCAP] ✅ 模型文件 '{fname}' 下载完成。")
                    break
                except Exception as e:
                    print(f"[AntiCAP] ⚠️ 下载尝试 {attempt} 失败: {e}")
                    if attempt < MAX_RETRIES:
                        print("[AntiCAP] 正在重试...")
                        time.sleep(5)
                    else:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        print(f"[AntiCAP] ❌ 模型文件 '{fname}' 下载失败，请手动下载并放置在 '{output_dir}'。")
                        print(f"[AntiCAP] 下载链接: https://github.com/81NewArk/AntiCAP/tree/main/AntiCAP/AntiCAP-Models")
                        raise IOError(f"无法下载模型文件 '{fname}'，请检查网络或稍后重试。")

    def get_onnx_session(self, model_path, use_gpu):
        key = (model_path, use_gpu)
        if key in self.models:
            return self.models[key]

        providers = ['CUDAExecutionProvider'] if use_gpu and onnxruntime.get_device().upper() == 'GPU' else [
            'CPUExecutionProvider']
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.models[key] = session
        return session

    def get_yolo_model(self, model_path, use_gpu):
        key = (model_path, use_gpu)
        if key in self.models:
            return self.models[key]

        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        model = YOLO(model_path, verbose=False)
        model.to(device)
        self.models[key] = model
        return model

    def get_charset(self, charset_path):
        if charset_path in self.charsets:
            return self.charsets[charset_path]
        
        try:
            with open(charset_path, 'r', encoding='utf-8') as f:
                list_as_string = f.read()
                charset = ast.literal_eval(list_as_string)
            self.charsets[charset_path] = charset
            return charset
        except FileNotFoundError:
            raise FileNotFoundError(f"字符集文件未在 {charset_path} 找到。")
        except Exception as e:
            raise ValueError(f"解析字符集文件时出错: {e}")


