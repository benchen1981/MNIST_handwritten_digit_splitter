import os
import cv2
import re
from .file_utils import ensure_folder

def infer_label_from_filename(filename: str) -> str:
    """
    嘗試從檔名中推斷數字標籤。
    策略：
    1. 移除 auto_split 或 grid_split 產生的後綴 (e.g., _digit_0, _0_1)。
    2. 在剩餘檔名中尋找獨立的數字 (0-9)。
    例如： "scan_smoothing_8_digit_0.png" -> "8"
    """
    name, _ = os.path.splitext(filename)
    
    # 1. 移除常見後綴 (由後往前匹配)
    # 移除 _digit_X (auto_split)
    name = re.sub(r'_digit_\d+$', '', name)
    # 移除 _X_X (grid_split)
    name = re.sub(r'_\d+_\d+$', '', name)

    # 2. 分割並尋找數字
    # 使用非數字字符分割，找出所有數字區塊
    parts = re.split(r'[^0-9]+', name)
    
    for p in parts:
        if p.isdigit():
             val = int(p)
             # MNIST 標籤應為 0-9
             if 0 <= val <= 9:
                 return str(val)
    
    return "unknown"

def classify_image(filename: str, input_dir: str, output_dir: str):
    """
    將 input_dir 中的 filename 讀取後，依照標籤存到 output_dir/label/ 下。
    """
    ensure_folder(output_dir)
    label = infer_label_from_filename(filename)
    save_dir = os.path.join(output_dir, label)
    ensure_folder(save_dir)

    src_path = os.path.join(input_dir, filename)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    dst_path = os.path.join(save_dir, filename)
    cv2.imwrite(dst_path, img)
