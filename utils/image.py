import math
import cv2
import numpy as np


def cylindrical_x(x: int, focal_length: float) -> float:
    return focal_length * math.atan(x / focal_length)


def cylindrical_y(x: int, y: int, focal_length: float) -> float:
    return focal_length * y / math.sqrt(x**2 + focal_length**2)


def build_intrinsic_mat(img: np.ndarray, focal_length: float) -> np.ndarray:
    h, w = img.shape[:2]
    return np.array(
        [[focal_length, 0, w / 2],
         [0, focal_length, h / 2],
         [0, 0, 1]]
    )


def to_cylindrical(img, focal_length: float):

    height, width = img.shape[:2]

    y_indices, x_indices = np.indices((height, width))

    k_mat = build_intrinsic_mat(img, focal_length)
    inv_k_mat = np.linalg.inv(k_mat)

    coords = np.stack([x_indices, y_indices, np.ones_like(x_indices)], axis=-1).reshape(height * width, 3)

    normed_coords = inv_k_mat.dot(coords.T).T

    cylindrical_coords = np.stack([np.sin(normed_coords[:, 0]), normed_coords[:, 1], np.cos(normed_coords[:, 0])], axis=-1).reshape(height * width, 3)

    coords_map = k_mat.dot(cylindrical_coords.T).T
    coords_map = coords_map[:, :-1] / coords_map[:, [-1]]
    coords_map = coords_map.reshape(height, width, -1)

    return cv2.remap(img, coords_map[:, :, 0].astype(np.float32), coords_map[:, :, 1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)


def x_blending(img: np.ndarray, start: int, end: int = -1) -> np.ndarray:

    width = img.shape[1]

    if end < 0:
        end = width

    interval = end - start

    start, end = min(start, end), max(start, end)

    x = np.zeros((width))
    alpha = 1 / abs(interval)
    if interval < 0:
        x[start:end] = np.arange(start=0, stop=1, step=alpha)
        x[end:] = 1
    else:
        x[start:end] = np.arange(start=1, stop=0, step=-alpha)
        x[:start] = 1

    img = img.astype(np.float64)

    img[:, :, :] *= x[np.newaxis, :, np.newaxis]

    return img.astype(np.uint8)


if __name__ == '__main__':

    filename = 'samples/0.jpeg'
    img = cv2.imread(filename)

    img = img[:, :, 0]

    warped_img = to_cylindrical(img, 1000)

    cv2.imshow("warped", warped_img)
    cv2.waitKey(0)
