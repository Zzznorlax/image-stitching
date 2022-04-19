from typing import List, Tuple
import cv2

from utils import image as img_utils
from utils import msop as msop_utils
from utils import file as file_utils


EXTS = [".jpg", ".jpeg", ".png"]


class ImageFragment():
    def __init__(self, img_path: str, focal_length: float = 800, layer_num: int = 1) -> None:

        self.img = cv2.imread(img_path)

        self.focal_length = focal_length

        self.cylindrical_img = img_utils.cylindrical_projection(self.img, self.focal_length)
        self.grayscale_img = cv2.cvtColor(self.cylindrical_img, cv2.COLOR_BGR2GRAY)

        print(self.grayscale_img.shape)

        self.resp_map = msop_utils.get_corner_resp_map(self.grayscale_img)

        self.hash_map = msop_utils.msop_detect(self.grayscale_img, layer_num)


class MatchResult():
    def __init__(self, fragment: ImageFragment, mv: Tuple[int, int]) -> None:
        self.fragment = fragment
        self.mv = mv


if __name__ == '__main__':

    img_folder = "samples/parrington"

    file_paths = file_utils.get_files(img_folder)

    img_frag_list: List[ImageFragment] = []
    for file_path in file_paths:
        _, ext = file_utils.get_extension(file_path)
        if ext not in EXTS:
            continue
        print(file_path)
        img_frag_list.append(ImageFragment(file_path))

    ordered_frags: List[ImageFragment] = []
    ordered_mvs = []
    for idx in range(len(img_frag_list)):

        frag = img_frag_list[idx]

        if not ordered_frags:
            ordered_frags.append(frag)
            continue

        # matches with the leftmost fragment
        mv_pairs = msop_utils.match(frag.hash_map, ordered_frags[0].hash_map)
        mv_pairs = msop_utils.mv_filter(frag.grayscale_img, mv_pairs)
        mv_pairs = msop_utils.reformat_mv()

        left_mv, left_inlier_count = msop_utils.ransac(mv_pairs)

        # matches with the rightmost fragment
        mv_pairs = msop_utils.match(frag.hash_map, ordered_frags[-1].hash_map)
        mv_pairs = msop_utils.mv_filter(frag.grayscale_img, mv_pairs)

        right_mv, right_inlier_count = msop_utils.ransac(mv_pairs)

        if right_inlier_count > left_inlier_count:
            ordered_frags.insert(0, frag)
            ordered_mvs.insert(0, right_mv)
        else:
            ordered_frags.append(frag)
            ordered_mvs.append(left_mv)

    result = ordered_frags[0].cylindrical_img
    for idx in range(1, len(ordered_frags)):
        result = msop_utils.blend_imgs(result, ordered_frags[idx].cylindrical_img, ordered_mvs[idx])

    cv2.imshow("Merged", result)
    cv2.waitKey(0)
