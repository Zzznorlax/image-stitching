
from typing import Tuple
from scipy.ndimage import gaussian_filter, maximum_filter

import numpy as np
import cv2


def get_smooth_grad(img: np.ndarray, sigma_d: float = 1) -> Tuple[np.ndarray, np.ndarray]:

    smoothed = gaussian_filter(img, sigma=sigma_d)

    grad_y, grad_x = np.gradient(smoothed)

    return grad_y, grad_x


def get_covariance_mat(img: np.ndarray, sigma_i: float = 1.5):

    grad_y, grad_x = get_smooth_grad(img)

    i_xx = gaussian_filter(grad_y * grad_y, sigma=sigma_i)
    i_xy = gaussian_filter(grad_x * grad_y, sigma=sigma_i)
    i_yy = gaussian_filter(grad_x * grad_x, sigma=sigma_i)

    h_mat = np.array([
        [i_xx, i_xy],
        [i_xy, i_yy]
    ])

    h_mat = np.transpose(h_mat, (2, 3, 0, 1))

    return h_mat


def get_corner_resp_map(img: np.ndarray, epsilon: float = 1e-8, local_maxima_size: int = 3, minimum: float = 10):

    hassian = get_covariance_mat(img)

    det = np.linalg.det(hassian)

    tr = np.trace(hassian, axis1=2, axis2=3) + epsilon

    keypoint = det / tr

    local_maxima = maximum_filter(keypoint, local_maxima_size)

    local_maxima[local_maxima < minimum] = 0

    return local_maxima


def msop_detect(img: np.ndarray, layer_num: int = 4, scale: int = 2, sigma_p: float = 1) -> np.ndarray:

    img = img.astype(np.float64)

    corner_map = np.zeros_like(img)

    for i in range(layer_num):

        layer_corner = get_corner_resp_map(img)

        corner_map[::scale**i, ::scale**i] += layer_corner

        smoothed = gaussian_filter(img, sigma_p)

        img = smoothed[::scale, ::scale]

    return corner_map


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    filename = 'samples/1.png'
    img = cv2.imread(filename)

    img_flipped = img[:, ::-1, 0]

    corner_flipped = msop_detect(img_flipped)

    img = img[:, :, 0]

    corner = msop_detect(img)

    imgs = [img, corner, img_flipped, corner_flipped]

    label_list = ['Original', 'keypoint', 'trace', 'det']
    for idx in range(len(imgs)):
        plt.subplot(2, 2, idx + 1)
        plt.title(label_list[idx])
        plt.imshow(imgs[idx], interpolation='none')
        plt.xticks([])
        plt.yticks([])

    # plt.subplot(2, 2, 4)
    # plt.title(label_list[3])
    # plt.quiver(grad_x, grad_y)
    # plt.xticks([])
    # plt.yticks([])

    plt.show()
