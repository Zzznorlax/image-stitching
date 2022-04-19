
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import gaussian_filter, maximum_filter, rotate

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

    def __init__(self, patches: Dict[int, List[OrientedPatch]], bin_num: int = 10) -> None:

        self.patch_count = 0
        self.patches = {}
        for layer_idx, layer_patches in patches.items():
            if layer_patches:
                self.patches[layer_idx] = layer_patches
                self.patch_count += len(layer_patches)

        self.hash_list: Dict[int, List[List[float]]] = {layer_idx: [list(patch.get_wavelet_hash()) for patch in patches] for layer_idx, patches in self.patches.items()}

        self.steps: Dict[int, Tuple[float, float, float]] = {layer_idx: ((max([hash[0] for hash in hashes]) - min([hash[0] for hash in hashes])) / bin_num,
                                                                         (max([hash[1] for hash in hashes]) - min([hash[1] for hash in hashes])) / bin_num,
                                                                         (max([hash[2] for hash in hashes]) - min([hash[2] for hash in hashes])) / bin_num) for layer_idx, hashes in self.hash_list.items()}

        self.lower_bounds: Dict[int, Tuple[float, float, float]] = {layer_idx: (min([hash[0] for hash in hashes]),
                                                                                min([hash[1] for hash in hashes]),
                                                                                min([hash[2] for hash in hashes])) for layer_idx, hashes in self.hash_list.items()}

        self.hash_map: Dict[int, Dict[Tuple[float, float, float], List[int]]] = {}

        for layer_idx, patch_list in self.patches.items():

            layer_hash_map = {}
            for patch_idx, patch in enumerate(patch_list):
                key = self.hash_key(patch, layer_idx)

                if key not in layer_hash_map:
                    layer_hash_map[key] = []

                layer_hash_map[key].append(patch_idx)

            self.hash_map[layer_idx] = layer_hash_map

    def hash_key(self, patch: OrientedPatch, layer: int) -> Tuple[float, float, float]:
        hash = patch.get_wavelet_hash()

        return tuple(((hash[i] - self.lower_bounds[layer][i]) // self.steps[layer][i]) for i in range(3))

    def match(self, patch: OrientedPatch, layer: int, thres: float = 1e-4) -> Optional[int]:
        if layer not in self.patches:
            return None

        key = self.hash_key(patch, layer)

        if key not in self.hash_map[layer]:
            return None

        match_idx = 0
        min_err = inf = 1e+8
        for idx in self.hash_map[layer][key]:
            patch_item = self.patches[layer][idx]
            mse = np.mean(np.square(patch.patch - patch_item.patch))

            if mse < min_err:
                match_idx = idx
                min_err = mse

        if min_err == inf or min_err > thres:
            return None

        return match_idx


def get_oriented_patches(img: np.ndarray, resp_map: np.ndarray, sigma_o: float = 4.5, sample_d: int = 5, radius: int = 20, sigma_p: float = 1, epsilon: float = 1e-8) -> List[OrientedPatch]:

    # Pₗ ∗ ∇σₒ(x, y)
    grad_y, grad_x = np.gradient(gaussian_filter(img, sigma=sigma_o))

    length = np.sqrt(grad_x**2 + grad_y**2) + epsilon

    grad_y /= length
    grad_x /= length

    sample_point_loc = np.where(resp_map > 0)

    # Pₗ ∗ gσₚ(x, y)
    smoothed = gaussian_filter(img, sigma=2 * sigma_p)

    height, width = img.shape

    patch_list = []

    for corner_idx in range(len(sample_point_loc[0])):
        corner_pos = (int(sample_point_loc[0][corner_idx]), int(sample_point_loc[1][corner_idx]))

        patch: np.ndarray = smoothed[max(0, corner_pos[0] - radius):min(height, corner_pos[0] + radius), max(0, corner_pos[1] - radius):min(width, corner_pos[1] + radius)]
        patch = patch[::sample_d, ::sample_d]

        if patch.size == 0 or patch.shape[0] != radius * 2 / sample_d or patch.shape[1] != radius * 2 / sample_d:
            continue

        rotated_patch = rotate(patch, angle=math.atan2(grad_x[corner_pos], grad_y[corner_pos]))

        # normalizes intensities
        rotated_patch = (rotated_patch - np.mean(rotated_patch)) / (np.var(rotated_patch) + epsilon)

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

    local_maxima = keypoint * (maximum_filter(keypoint, local_maxima_size) == keypoint)

    local_maxima[local_maxima < minimum] = 0

    return local_maxima


def match(map_a: WaveletHashmap, map_b: WaveletHashmap):
    mv_pairs = []

    for layer_idx, patches in map_b.patches.items():
        for patch in patches:
            matched_idx = map_a.match(patch, layer=layer_idx)
            if matched_idx is None:
                continue

            matched_patch = map_a.patches[layer_idx][matched_idx]

            y_a = matched_patch.pos[0] * (2**layer_idx)
            x_a = matched_patch.pos[1] * (2**layer_idx)

            y_b = patch.pos[0] * (2**layer_idx)
            x_b = patch.pos[1] * (2**layer_idx)

            mv_pairs.append([y_a, x_a, y_b, x_b])

    return mv_pairs


def ransac(mv_pairs: List, k: int = 2000, n: int = 2, diff_thres: float = 5) -> Tuple[Tuple[float, float], int]:

    num_pairs = len(mv_pairs)

    pairs = np.array(mv_pairs).astype(np.float32)
    diff = np.empty((num_pairs, 2))

    max_inlier = 0
    best_mv = (0, 0)
    for _ in range(k):

        samples = pairs[np.random.choice(num_pairs, n)]

        sample_dy = np.sum(samples[:, 0] - samples[:, 2]) / n
        sample_dx = np.sum(samples[:, 1] - samples[:, 3]) / n

        diff[:, 0] = np.absolute(pairs[:, 0] - pairs[:, 2] - sample_dy)
        diff[:, 1] = np.absolute(pairs[:, 1] - pairs[:, 3] - sample_dx)

        inlier_indices = np.where((diff < diff_thres).all(axis=1))[0]

        inlier_count = len(inlier_indices)

        if inlier_count > max_inlier:
            max_inlier = inlier_count

            inlier_mv = np.zeros((num_pairs, 2))
            inlier_mv[:, 0] = pairs[:, 0] - pairs[:, 2]
            inlier_mv[:, 1] = pairs[:, 1] - pairs[:, 3]

            outlier_indices = np.where((inlier_mv > diff_thres).all(axis=1))
            inlier_mv[outlier_indices[0], :] = 0

            best_mv = (float(np.average(inlier_mv[inlier_indices, 0])), float(np.average(inlier_mv[inlier_indices, 1])))

    return best_mv, max_inlier


def reformat_mv(img_a: np.ndarray, img_b: np.ndarray, mv_mat: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:

    img_a = img_a.astype(np.float64)
    img_b = img_b.astype(np.float64)

    mv_y, mv_x = mv_mat

    rounded_mv_x = math.floor(abs(mv_x))
    if mv_x < 0:
        mv_y = -mv_y
        img_a, img_b = img_b, img_a

    rounded_mv_y = math.floor(abs(mv_y))
    if mv_y < 0:
        rounded_mv_y = -rounded_mv_y

    _, width = img_a.shape[:2]

    return img_a, img_b, (rounded_mv_y, width - rounded_mv_x)


def blend_imgs(img_a: np.ndarray, img_b: np.ndarray, mv_mat: Tuple[int, int]):

    channel = 0
    if len(img_a.shape) == 3:
        channel = img_a.shape[2]

    height_a, width_a = img_a.shape[:2]
    height_b, width_b = img_b.shape[:2]

    mv_y, mv_x = mv_mat

    img_a = img_a.astype(np.float64)
    img_b = img_b.astype(np.float64)

    abs_mv_y = abs(mv_y)

    result_width = width_a + width_b - mv_x
    result_height = max(height_a, height_b) + abs_mv_y
    if abs(height_a - height_b) > abs_mv_y:
        result_height = max(height_a, height_b)

    result = np.zeros((result_height + abs_mv_y, result_width, channel), dtype=np.float64)

    img_a = img_utils.x_blending(img_a, width_a - mv_x, width_a)
    img_b = img_utils.x_blending(img_b, mv_x, 0)

    if mv_y > 0:
        result[:height_a, :width_a] += img_a
        result[-height_b:, -width_b:] += img_b

    else:
        result[-height_a:, :width_a] += img_a
        result[:height_b, -width_b:] += img_b

    return result.astype(np.uint8)


def mv_filter(img: np.ndarray, mv_pairs: List[List[int]], y_range_thres: float = 0.1) -> List[List[int]]:

    height, width = img.shape[:2]

    filtered = []
    for mv_pair in mv_pairs:
        y_a, x_a, y_b, x_b = mv_pair
        if abs(y_a - y_b) < height * y_range_thres:
            filtered.append(mv_pair)

    return filtered


def nms(resp_map: np.ndarray, n: int = 10) -> np.ndarray:
    return resp_map * (maximum_filter(resp_map, n) == resp_map)


def msop_detect(img: np.ndarray, layer_num: int = 4, scale: int = 2, sigma_p: float = 1) -> WaveletHashmap:

    img = img.astype(np.float64)

    img_layer = img

    patches: Dict[int, List[OrientedPatch]] = {}

    for layer_idx in range(layer_num):

        resp_map = get_corner_resp_map(img_layer)

        resp_map = nms(resp_map, n=20)

        patches[layer_idx] = get_oriented_patches(img_layer, resp_map)

        smoothed = gaussian_filter(img_layer, sigma_p)
        img_layer = smoothed[::scale, ::scale]

    return WaveletHashmap(patches)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    filename = 'samples/1.png'
    img = cv2.imread(filename)

    img = img[:, :, 0]
    img_flipped = img[:, ::-1, 0]

    blended = blend_imgs(img, img_flipped, (100, 10))

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
