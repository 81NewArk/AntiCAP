# coding=utf-8

import logging
from .utils.manager import ModelManager
from .modules import ocr, math_solver, detection, click, slider, similarity, rotation

class Handler(object):
    logging.getLogger('ultralytics').setLevel(logging.WARNING)

    def __init__(self, show_banner=True):
        self.manager = ModelManager()
        
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
            |         Blog  : https://www.newark81.vip/               |          
            |         Author: 81NewArk                                |       
            -----------------------------------------------------------''')

    # 文字识别
    def OCR(self, img_base64: str = None, use_gpu: bool = False, png_fix: bool = False, probability=False):
        return ocr.solve_ocr(self.manager, img_base64, use_gpu, png_fix, probability)

    # 算术识别
    def Math(self, img_base64: str, math_model_path: str = '', use_gpu: bool = False):
        return math_solver.solve_math(self.manager, img_base64, math_model_path, use_gpu)

    # 图标侦测
    def Detection_Icon(self, img_base64: str = None, detectionIcon_model_path: str = '', use_gpu: bool = False):
        return detection.solve_detection_icon(self.manager, img_base64, detectionIcon_model_path, use_gpu)

    # 按序侦测图标
    def ClickIcon_Order(self, order_img_base64: str, target_img_base64: str, detectionIcon_model_path: str = '', sim_onnx_model_path: str = '', use_gpu: bool = False):
        return click.solve_click_icon_order(self.manager, order_img_base64, target_img_base64, detectionIcon_model_path, sim_onnx_model_path, use_gpu)

    # 文字侦测
    def Detection_Text(self, img_base64: str = None, detectionText_model_path: str = '', use_gpu: bool = False):
        return detection.solve_detection_text(self.manager, img_base64, detectionText_model_path, use_gpu)

    # 按序侦测文字
    def ClickText_Order(self, order_img_base64: str, target_img_base64: str, detectionText_model_path: str = '', sim_onnx_model_path: str = '', use_gpu: bool = False):
        return click.solve_click_text_order(self.manager, order_img_base64, target_img_base64, detectionText_model_path, sim_onnx_model_path, use_gpu)

    # 缺口滑块
    def Slider_Match(self, target_base64: str = None, background_base64: str = None, simple_target: bool = False, flag: bool = False):
        return slider.solve_slider_match(self.manager, target_base64, background_base64, simple_target, flag)

    # 阴影滑块
    def Slider_Comparison(self, target_base64: str = None, background_base64: str = None):
        return slider.solve_slider_comparison(self.manager, target_base64, background_base64)

    # 图像相似度比较
    def Compare_Image_Similarity(self, image1_base64: str, image2_base64: str, sim_onnx_model_path: str = None, use_gpu: bool = False):
        return similarity.solve_compare_image_similarity(self.manager, image1_base64, image2_base64, sim_onnx_model_path, use_gpu)

    # 单图旋转角度
    def Single_Rotate(self, img_base64: str, rotate_onnx_modex_path: str = '', use_gpu: bool = False):
        return rotation.solve_single_rotate(self.manager, img_base64, rotate_onnx_modex_path, use_gpu)

    # 双图旋转
    def Double_Rotate(self, inside_base64: str, outside_base64: str, check_pixel: int = 10, speed_ratio: float = 1, grayscale: bool = False, anticlockwise: bool = False, cut_pixel_value: int = 0):
        return rotation.solve_double_rotate(self.manager, inside_base64, outside_base64, check_pixel, speed_ratio, grayscale, anticlockwise, cut_pixel_value)
