import numpy as np
from scipy.optimize import linear_sum_assignment
from .similarity import get_siamese_similarity
from ..utils.common import get_model_path, decode_base64_to_image


def handle_matching(manager, order_image, target_image, order_boxes_list, target_boxes_list, siamese_model_path, use_gpu, sim_onnx_model_path):
    """
    通用匹配处理函数，使用匈牙利算法（Hungarian Algorithm）求解全局最优匹配
    """
    if not order_boxes_list or not target_boxes_list:
        return [[0, 0, 0, 0]] * len(order_boxes_list)

    num_orders = len(order_boxes_list)
    num_targets = len(target_boxes_list)
    
    # 1. 构建相似度矩阵 (Cost Matrix)
    # 我们希望总相似度最大，linear_sum_assignment 是求最小开销，所以存入负的相似度
    cost_matrix = np.zeros((num_orders, num_targets))

    for i, order_box in enumerate(order_boxes_list):
        order_crop = order_image.crop(order_box)
        if order_crop.width == 0 or order_crop.height == 0:
            continue
            
        for j, target_box in enumerate(target_boxes_list):
            target_crop = target_image.crop(target_box)
            if target_crop.width == 0 or target_crop.height == 0:
                continue
                
            similarity_score = get_siamese_similarity(
                manager, order_crop, target_crop, 
                siamese_model_path, use_gpu, 
                is_custom_model=bool(sim_onnx_model_path)
            )
            # 记录分值 (由于求解器求最小，我们存负值)
            cost_matrix[i, j] = -similarity_score

    # 2. 执行匈牙利算法求解
    # row_ind 会对应 order_boxes_list 的索引，col_ind 会对应最优匹配的 target_boxes_list 索引
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 3. 构造返回结果，保持与 order_boxes_list 顺序一致
    # 初始化全 0 结果
    best_matching_boxes = [[0, 0, 0, 0]] * num_orders
    
    # 填充结果
    # 注意：如果目标比候选框多，这里的 row_ind/col_ind 只会包含匹配上的部分
    for r, c in zip(row_ind, col_ind):
        # 即使匹配上了，如果得分太低（例如相似度为 0），也可以视作无效匹配
        # 这里保留原逻辑的灵活性，目前只要有匹配索引就写入
        target_box = target_boxes_list[c]
        best_matching_boxes[r] = [int(coord) for coord in target_box]

    return best_matching_boxes


def solve_click_icon_order(manager, order_img_base64: str, target_img_base64: str, detectionIcon_model_path: str = '', sim_onnx_model_path: str = '', use_gpu: bool = False):
    detectionIcon_model_path = detectionIcon_model_path or get_model_path('[AntiCAP]-Detection_Icon-YOLO.pt')
    
    if sim_onnx_model_path:
        siamese_model_path = sim_onnx_model_path
    else:
        siamese_model_path = get_model_path('[AntiCAP]-Siamese-ResNet18.onnx')

    model = manager.get_yolo_model(detectionIcon_model_path, use_gpu)

    order_image = decode_base64_to_image(order_img_base64).convert("RGB")
    target_image = decode_base64_to_image(target_img_base64).convert("RGB")

    order_results = model(order_image, verbose=False)
    target_results = model(target_image, verbose=False)

    order_boxes_list = []
    if order_results and order_results[0].boxes:
        order_boxes = order_results[0].boxes.xyxy.cpu().numpy().tolist()
        order_boxes.sort(key=lambda x: x[0])  # 按提示图的 X 坐标排序（通常文字提示是从左到右的）
        order_boxes_list = order_boxes

    target_boxes_list = []
    if target_results and target_results[0].boxes:
        target_boxes_list = target_results[0].boxes.xyxy.cpu().numpy().tolist()

    return handle_matching(manager, order_image, target_image, order_boxes_list, target_boxes_list, siamese_model_path, use_gpu, sim_onnx_model_path)


def solve_click_text_order(manager, order_img_base64: str, target_img_base64: str, detectionText_model_path: str = '', sim_onnx_model_path: str = '', use_gpu: bool = False):
    detectionText_model_path = detectionText_model_path or get_model_path('[AntiCAP]-Detection_Text-YOLO.pt')
    
    if sim_onnx_model_path:
        siamese_model_path = sim_onnx_model_path
    else:
        siamese_model_path = get_model_path('[AntiCAP]-Siamese-ResNet18.onnx')

    model = manager.get_yolo_model(detectionText_model_path, use_gpu)

    order_image = decode_base64_to_image(order_img_base64).convert("RGB")
    target_image = decode_base64_to_image(target_img_base64).convert("RGB")

    order_results = model(order_image, verbose=False)
    target_results = model(target_image, verbose=False)

    order_boxes_list = []
    if order_results and order_results[0].boxes:
        order_boxes = order_results[0].boxes.xyxy.cpu().numpy().tolist()
        order_boxes.sort(key=lambda x: x[0])
        order_boxes_list = order_boxes

    target_boxes_list = []
    if target_results and target_results[0].boxes:
        target_boxes_list = target_results[0].boxes.xyxy.cpu().numpy().tolist()

    return handle_matching(manager, order_image, target_image, order_boxes_list, target_boxes_list, siamese_model_path, use_gpu, sim_onnx_model_path)
