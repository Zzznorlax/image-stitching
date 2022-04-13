
from typing import List, Optional, Tuple
from scipy.ndimage import gaussian_filter, maximum_filter

import math
import numpy as np
import cv2

from utils import image as img_utils


class OrientedPatch():
    def __init__(self, pos: Tuple[int, int], patch: np.ndarray) -> None:
        self.pos = pos
        self.patch = patch

        self.wavelet_hash: Optional[Tuple[float, float, float]] = None

    def get_wavelet_hash(self) -> Tuple[float, float, float]:

        if self.wavelet_hash is None:

            descriptor_0 = -np.sum(self.patch[:, :-(self.patch.shape[1] // 2)]) + np.sum(self.patch[:, self.patch.shape[1] // 2:])

            descriptor_1 = np.sum(self.patch[:-(self.patch.shape[0] // 2), :]) - np.sum(self.patch[self.patch.shape[0] // 2:, :])

            descriptor_2 = -np.sum(self.patch[:-(self.patch.shape[0] // 2), :-(self.patch.shape[1] // 2)]) - np.sum(self.patch[self.patch.shape[0] // 2:, self.patch.shape[1] // 2:])
            descriptor_2 += np.sum(self.patch[:-(self.patch.shape[0] // 2), self.patch.shape[1] // 2:]) + np.sum(self.patch[self.patch.shape[0] // 2:, :-(self.patch.shape[1] // 2)])

            self.wavelet_hash = (descriptor_0, descriptor_1, descriptor_2)

        return self.wavelet_hash


class WaveletHashmap():

    def __init__(self, patch_list: List[OrientedPatch]) -> None:

        self.patch_list = patch_list

        self.hash_list: List[List[float]] = [list(patch.get_wavelet_hash()) for patch in self.patch_list]

        self.upper_bounds = (max([hash[0] for hash in self.hash_list]), max([hash[1] for hash in self.hash_list]), max([hash[2] for hash in self.hash_list]))
        self.lower_bounds = (min([hash[0] for hash in self.hash_list]), min([hash[1] for hash in self.hash_list]), min([hash[2] for hash in self.hash_list]))

        self.steps = [(self.upper_bounds[0] - self.lower_bounds[0]) / 10, (self.upper_bounds[1] - self.lower_bounds[1]) / 10, (self.upper_bounds[2] - self.lower_bounds[2]) / 10]

        self.hash_map = {}

        for idx, patch in enumerate(self.patch_list):
            key = self.hash_key(patch)

            if key not in self.hash_map:
                self.hash_map[key] = []

            self.hash_map[key].append(idx)

    def hash_key(self, patch: OrientedPatch) -> Tuple[int, int, int]:
        hash = patch.get_wavelet_hash()

        return tuple(int((hash[i] - self.lower_bounds[i]) // self.steps[i]) for i in range(3))

    def match(self, patch: OrientedPatch) -> Optional[int]:
        key = self.hash_key(patch)

        if key not in self.hash_map:
            return None

        match_idx = 0
        min_err = 0
        for idx in self.hash_map[key]:
            patch_item = self.patch_list[idx]
            mse = np.square(patch.patch - patch_item.patch).mean()  # type: ignore

            if mse < min_err:
                match_idx = idx
                min_err = mse

        return match_idx


def get_oriented_patches(img: np.ndarray, resp_map: np.ndarray, sigma_o: float = 4.5, sample_d: int = 5, radius: int = 20, sigma_p: float = 1) -> List[OrientedPatch]:

    # Pₗ ∗ ∇σₒ(x, y)
    grad_y, grad_x = np.gradient(gaussian_filter(img, sigma=sigma_o))

    length = np.sqrt(grad_x**2 + grad_y**2)

    grad_y /= length
    grad_x /= length

    sample_point_loc = np.where(resp_map > 0)

    # Pₗ ∗ gσₚ(x, y)
    smoothed = gaussian_filter(img, sigma=2 * sigma_p)

    height, width = img.shape

    patch_list = []

    for corner_idx in range(len(sample_point_loc[0])):
        corner_pos = (sample_point_loc[0][corner_idx], sample_point_loc[1][corner_idx])

        patch = smoothed[max(0, corner_pos[0] - radius):min(height, corner_pos[1] + radius), max(0, corner_pos[1] - radius):min(width, corner_pos[1] + radius)]
        patch = patch[::sample_d, ::sample_d]
        rotation_mat = cv2.getRotationMatrix2D(corner_pos, math.atan2(grad_x[corner_pos], grad_y[corner_pos]), scale=1.0)
        rotated_patch = cv2.warpAffine(patch, rotation_mat, (patch.shape[1], patch.shape[0]))

        # normalizes intensities
        rotated_patch = (rotated_patch - np.mean(rotated_patch)) / np.var(rotated_patch)

        patch_list.append(OrientedPatch(corner_pos, rotated_patch))

    return patch_list


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


def match_patches(map_a: WaveletHashmap, map_b: WaveletHashmap, layer: int = 0) -> List:
    mv_pairs = []

    for patch in map_b.patch_list:
        matched_idx = map_a.match(patch)
        if matched_idx is None:
            continue

        matched_patch = map_a.patch_list[matched_idx]

        y_a = matched_patch.pos[0] * (2**layer)
        x_a = matched_patch.pos[1] * (2**layer)

        y_b = patch.pos[0] * (2**layer)
        x_b = patch.pos[1] * (2**layer)

        mv_pairs.append([y_a, x_a, y_b, x_b])

    return mv_pairs


def ransac(mv_pairs: List, k: int = 60, n: int = 2, thres: float = 5):

    num_pairs = len(mv_pairs)

    pairs = np.array(mv_pairs)
    diff = np.empty((num_pairs, 2))

    max_inlier = 0
    best_mv = None
    for _ in range(k):

        samples = pairs[np.random.choice(num_pairs, n)]

        sample_dy = np.sum(samples[:, 0] - samples[:, 2]) / n
        sample_dx = np.sum(samples[:, 1] - samples[:, 3]) / n

        diff[:, 0] = np.absolute(samples[:, 2] - samples[:, 0] + sample_dy)
        diff[:, 1] = np.absolute(samples[:, 1] - samples[:, 3] + sample_dx)

        inlier_count = np.sum(np.where((diff < thres).all(axis=1)))
        if inlier_count > max_inlier:
            max_inlier = inlier_count
            best_mv = (sample_dy, sample_dx)

    return best_mv


def blending(img_a: np.ndarray, img_b: np.ndarray, mv_mat: Tuple[float, float]):

    channel = 0
    if len(img_a.shape) == 3:
        channel = img_a.shape[2]

    height, width = img_a.shape[:2]
    mv_y, mv_x = mv_mat

    img_a = img_a.astype(np.float64)
    img_b = img_b.astype(np.float64)

    if mv_x < 0:
        img_a, img_b = img_b, img_a
        mv_x = -mv_x
        mv_y = -mv_y

    mv_x = math.floor(mv_x)
    abs_mv_y = math.floor(abs(mv_y))

    result = np.zeros((height + abs_mv_y, width + mv_x, channel), dtype=np.float64)

    img_a = img_utils.x_blending(img_a, width - mv_x)
    img_b = img_utils.x_blending(img_b, mv_x, 0)

    if mv_y > 0:
        result[:height, :width] += img_a
        result[abs_mv_y:, mv_x:] += img_b
    else:
        result[abs_mv_y:, :width] += img_a
        result[:height, mv_x:] += img_b

    return result.astype(np.uint8)


def msop_detect(img: np.ndarray, layer_num: int = 4, scale: int = 2, sigma_p: float = 1) -> List[List[OrientedPatch]]:

    img = img.astype(np.float64)

    patches_pyramid = []

    img_layer = img
    for i in range(layer_num):

        resp_map = get_corner_resp_map(img_layer)

        patches = get_oriented_patches(img_layer, resp_map)

        patches_pyramid.append(patches)

        smoothed = gaussian_filter(img_layer, sigma_p)
        img_layer = smoothed[::scale, ::scale]

    return patches_pyramid


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    filename = 'samples/1.png'
    img = cv2.imread(filename)

    img = img[:, :, 0]
    img_flipped = img[:, ::-1, 0]

    blended = blending(img, img_flipped, (100, 10))

    cv2.imshow("blended", blended)
    cv2.waitKey(0)

    # corner_flipped = resp_map = get_corner_resp_map(img_flipped)

    # img = img[:, :, 0]

    # corner = get_corner_resp_map(img)

    # imgs = [img, corner, img_flipped, corner_flipped]

    # label_list = ['Original', 'keypoint', 'trace', 'det']
    # for idx in range(len(imgs)):
    #     plt.subplot(2, 2, idx + 1)
    #     plt.title(label_list[idx])
    #     plt.imshow(imgs[idx], interpolation='none')
    #     plt.xticks([])
    #     plt.yticks([])

    # plt.show()
