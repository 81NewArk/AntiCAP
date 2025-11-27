from .similarity import get_siamese_similarity

from ..utils.common import get_model_path, decode_base64_to_image

def solve_click_icon_order(manager, order_img_base64: str, target_img_base64: str, detectionIcon_model_path: str = '', siamese_model_path: str = '', use_gpu: bool = False):
    detectionIcon_model_path = detectionIcon_model_path or get_model_path('[AntiCAP]-Detection_Icon-YOLO.pt')
    siamese_model_path = siamese_model_path or get_model_path('[AntiCAP]-Siamese-ResNet18.onnx')

    model = manager.get_yolo_model(detectionIcon_model_path, use_gpu)

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

    available_target_boxes = target_boxes_list.copy()
    best_matching_boxes = []

    for order_box in order_boxes_list:
        order_crop = order_image.crop(order_box)
        if order_crop.width == 0 or order_crop.height == 0:
            best_matching_boxes.append([0, 0, 0, 0])
            continue

        best_score = -1
        best_target_box = None

        for target_box in available_target_boxes:
            target_crop = target_image.crop(target_box)
            if target_crop.width == 0 or target_crop.height == 0:
                continue

            similarity_score = get_siamese_similarity(manager, order_crop, target_crop, siamese_model_path, use_gpu)

            if similarity_score > best_score:
                best_score = similarity_score
                best_target_box = target_box

        if best_target_box:
            best_matching_boxes.append([int(coord) for coord in best_target_box])
            available_target_boxes.remove(best_target_box)
        else:
            best_matching_boxes.append([0, 0, 0, 0])

    return best_matching_boxes

def solve_click_text_order(manager, order_img_base64: str, target_img_base64: str, detectionText_model_path: str = '', siamese_model_path: str = '', use_gpu: bool = False):
    detectionText_model_path = detectionText_model_path or get_model_path('[AntiCAP]-Detection_Text-YOLO.pt')
    siamese_model_path = siamese_model_path or get_model_path('[AntiCAP]-Siamese-ResNet18.onnx')

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

    available_target_boxes = target_boxes_list.copy()
    best_matching_boxes = []

    for order_box in order_boxes_list:
        order_crop = order_image.crop(order_box)
        if order_crop.width == 0 or order_crop.height == 0:
            best_matching_boxes.append([0, 0, 0, 0])
            continue

        best_score = -1
        best_target_box = None

        for target_box in available_target_boxes:
            target_crop = target_image.crop(target_box)
            if target_crop.width == 0 or target_crop.height == 0:
                continue

            similarity_score = get_siamese_similarity(manager, order_crop, target_crop, siamese_model_path, use_gpu)

            if similarity_score > best_score:
                best_score = similarity_score
                best_target_box = target_box

        if best_target_box:
            best_matching_boxes.append([int(coord) for coord in best_target_box])
            available_target_boxes.remove(best_target_box)
        else:
            best_matching_boxes.append([0, 0, 0, 0])

    return best_matching_boxes
