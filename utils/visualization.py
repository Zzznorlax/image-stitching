import cv2
import numpy as np


def draw_line(img: np.ndarray, p1, p2, color=(0, 255, 0)):
    cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color=color, thickness=1)


def draw_dot(img: np.ndarray, x, y, radius=2, color=(0, 0, 255)):
    cv2.circle(img, (int(x), int(y)), radius=radius, color=color, thickness=-1)


def show_resp(img: np.ndarray, resp_map: np.ndarray) -> np.ndarray:

    y_indices, x_indices = np.where(resp_map > 0)

    for i in range(len(y_indices)):
        draw_dot(img, x_indices[i], y_indices[i])

    return img


def show_matched(img_1, img_2, pos_pairs) -> np.ndarray:
    height_1, width_1 = img_1.shape[:2]
    height_2, width_2 = img_2.shape[:2]

    result = np.empty((max(height_1, height_2), width_1 + width_2, 3), dtype=np.float32)

    result[:, :width_1] = img_1.astype(np.float32)
    result[:, -width_2:] = img_2.astype(np.float32)

    for pos_pair in pos_pairs:
        draw_line(result, (pos_pair[1], pos_pair[0]), (pos_pair[3] + width_1, pos_pair[2]))

    return result.astype(np.uint8)
