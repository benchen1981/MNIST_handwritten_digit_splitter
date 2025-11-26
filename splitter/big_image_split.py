import os
import cv2
import numpy as np
from .file_utils import ensure_folder

def preprocess_for_digits(img_gray):
    """
    將灰階圖做二值化 + 形態學處理，讓數字輪廓更清楚。
    """
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh

def resize_to_mnist_style(crop, target_size=28, inner_size=20):
    """
    將裁出來的數字圖 resize 成類似 MNIST 的格式。
    """
    h, w = crop.shape
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    if h > w:
        new_h = inner_size
        new_w = int(w * inner_size / h)
    else:
        new_w = inner_size
        new_h = int(h * inner_size / w)

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

def auto_split_digits(image_path: str, save_dir: str, min_area: int = 50, return_boxes=False):
    """
    自動偵測並切割大圖中的每一個數字，存成獨立影像。
    可選擇是否回傳 bounding boxes 與 digits。
    """
    ensure_folder(save_dir)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        raise ValueError(f"無法讀取影像：{image_path}")

    bin_img = preprocess_for_digits(img_gray)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        bboxes.append((x, y, w, h))

    if not bboxes:
        if return_boxes:
            return 0, [], []
        return 0

    bboxes.sort(key=lambda b: (b[0], b[1]))

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    count = 0
    digits = []

    for idx, (x, y, w, h) in enumerate(bboxes):
        digit_crop = img_gray[y:y + h, x:x + w]
        digit_norm = resize_to_mnist_style(digit_crop, 28, 20)

        out_name = f"{base_name}_digit_{idx}.png"
        out_path = os.path.join(save_dir, out_name)
        cv2.imwrite(out_path, digit_norm)
        digits.append(digit_norm)
        count += 1

    if return_boxes:
        return count, bboxes, digits
    return count

def grid_split_image(image_path: str, save_dir: str, grid_cols: int = 10, grid_rows: int = 1):
    """
    傳統格子等分切割版本（備援使用）。
    """
    ensure_folder(save_dir)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"無法讀取影像：{image_path}")

    h, w = img.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    count = 0
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for r in range(grid_rows):
        for c in range(grid_cols):
            sub_img = img[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            out_name = f"{base_name}_{r}_{c}.png"
            out_path = os.path.join(save_dir, out_name)
            cv2.imwrite(out_path, sub_img)
            count += 1

    return count
