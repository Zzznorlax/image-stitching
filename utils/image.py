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


def to_cylindrical(img, focal_length: float = 800):

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

    mapped = cv2.remap(img, coords_map[:, :, 0].astype(np.float32), coords_map[:, :, 1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

    return mapped


def x_blending(img: np.ndarray, start: int, end: int = -1) -> np.ndarray:

    width = img.shape[1]

    if end < 0:
        end = width

    interval = end - start

    if interval == 0:
        return img

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


def cylindrical_projection(img, focal_length):

    height, width = img.shape[:2]

    cylinder_proj = np.zeros(shape=img.shape, dtype=np.uint8)

    for y in range(-int(height / 2), int(height / 2)):
        for x in range(-int(width / 2), int(width / 2)):
            cylinder_x = focal_length * math.atan(x / focal_length)
            cylinder_y = focal_length * y / math.sqrt(x**2 + focal_length**2)

            cylinder_x = round(cylinder_x + width / 2)
            cylinder_y = round(cylinder_y + height / 2)

            if cylinder_x >= 0 and cylinder_x < width and cylinder_y >= 0 and cylinder_y < height:
                cylinder_proj[cylinder_y][cylinder_x] = img[y + int(height / 2)][x + int(width / 2)]

    # Crop black border
    # ref: http://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    if(len(img.shape) > 2):
        _, thresh = cv2.threshold(cv2.cvtColor(cylinder_proj, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(cylinder_proj, 1, 255, cv2.THRESH_BINARY)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0][0])

    return cylinder_proj[y:y + h, x:x + w]


if __name__ == '__main__':

    filename = 'samples/0.jpeg'
    img = cv2.imread(filename)

    img = img[:, :, 0]

    warped_img = to_cylindrical(img, 1000)

    cv2.imshow("warped", warped_img)
    cv2.waitKey(0)
