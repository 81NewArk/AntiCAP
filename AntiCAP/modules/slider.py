import io
import cv2
import numpy as np
from PIL import Image, ImageChops
from ..utils.common import decode_base64_to_cv2, decode_base64_to_image

def solve_slider_match(manager, target_base64: str = None, background_base64: str = None, simple_target: bool = False, flag: bool = False):
    
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



    if not simple_target:
        target_image = decode_base64_to_cv2(target_base64)
        if target_image is None:
            if flag:
                raise ValueError("Failed to decode target base64 image.")
            return solve_slider_match(manager, target_base64=target_base64,
                                     background_base64=background_base64,
                                     simple_target=True, flag=True)
        try:
            target_pil, target_x, target_y = get_target(target_image.tobytes())
            if target_pil is None:
                if flag:
                    raise ValueError("Failed to extract target from image.")
                return solve_slider_match(manager, target_base64=target_base64,
                                         background_base64=background_base64,
                                         simple_target=True, flag=True)
            target = cv2.cvtColor(np.asarray(target_pil), cv2.COLOR_RGB2BGR)
        except SystemError as e:
            if flag:
                raise e
            return solve_slider_match(manager, target_base64=target_base64,
                                     background_base64=background_base64,
                                     simple_target=True, flag=True)
    else:
        target = decode_base64_to_cv2(target_base64)
        if target is None:
            return {"target": [0, 0, 0, 0]}
        target_y = 0
        target_x = 0

    background = decode_base64_to_cv2(background_base64)
    if background is None:
        return {"target": [0, 0, 0, 0]}

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

    return {"target": [int(max_loc[0]), int(max_loc[1]), int(bottom_right[0]), int(bottom_right[1])]}

def solve_slider_comparison(manager, target_base64: str = None, background_base64: str = None):
    target = decode_base64_to_image(target_base64).convert("RGB")
    background = decode_base64_to_image(background_base64).convert("RGB")

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
