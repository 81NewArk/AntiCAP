from ..utils.common import get_model_path, decode_base64_to_image

def solve_detection_icon(manager, img_base64: str = None, detectionIcon_model_path: str = '', use_gpu: bool = False):
    detectionIcon_model_path = detectionIcon_model_path or get_model_path('[AntiCAP]-Detection_Icon-YOLO.pt')
    model = manager.get_yolo_model(detectionIcon_model_path, use_gpu)

    image = decode_base64_to_image(img_base64)

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

def solve_detection_text(manager, img_base64: str = None, detectionText_model_path: str = '', use_gpu: bool = False):
    detectionText_model_path = detectionText_model_path or get_model_path('[AntiCAP]-Detection_Text-YOLO.pt')
    model = manager.get_yolo_model(detectionText_model_path, use_gpu)

    image = decode_base64_to_image(img_base64).convert("RGB")

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
