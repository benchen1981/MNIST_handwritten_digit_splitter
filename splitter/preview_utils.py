import cv2
import numpy as np

def draw_bounding_boxes(original_img, bboxes, color=(0, 255, 0)):
    """
    在原圖上把輪廓框畫出來。
    """
    img_color = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in bboxes:
        cv2.rectangle(img_color, (x, y), (x + w, y + h), color, 2)
    return img_color

def enhance_digit(digit):
    """
    做視覺調整：對比增強 + 邊緣強化。
    """
    img = digit.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img = clahe.apply(img)

    sharp_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    img = cv2.filter2D(img, -1, sharp_kernel)
    return img

def gray_profile(img):
    """
    取得水平與垂直投影（灰階 profile）。
    """
    vertical = np.sum(img, axis=0)
    horizontal = np.sum(img, axis=1)
    return horizontal, vertical
