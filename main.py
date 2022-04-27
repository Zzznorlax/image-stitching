from typing import List
from os.path import join
import argparse
import cv2
import numpy as np

from utils import image as img_utils
from utils import msop as msop_utils
from utils import file as file_utils
from utils import log as log_utils


EXTS = [".jpg", ".jpeg", ".png"]
DEFAULT_DIR = "samples/data"
OUTPUT_FILENAME = "output.jpg"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=DEFAULT_DIR, help='input images directory')
    parser.add_argument('--output', type=str, nargs='?', const=OUTPUT_FILENAME, help='output image path')
    parser.add_argument('--inlier-thres', type=float, default=0, help='inlier threshold')
    parser.add_argument('--fc', type=float, default=800, help='focal length')

    opt = parser.parse_args()

    return opt


class ImageFragment():
    def __init__(self, img_path: str, focal_length: float = 2000, layer_num: int = 1) -> None:

        self.path = img_path

        self.layer_num = layer_num

        self.img = cv2.imread(img_path)

        self.focal_length = focal_length

        self.cylindrical_img = img_utils.cylindrical_projection(self.img, self.focal_length)

        self.grayscale_img = cv2.cvtColor(self.cylindrical_img, cv2.COLOR_BGR2GRAY)

        self.resp_map = msop_utils.get_corner_resp_map(self.grayscale_img)

        self.hash_map = msop_utils.msop_detect(self.grayscale_img, layer_num)

    def replace(self, img: np.ndarray):
        self.cylindrical_img = img

        self.grayscale_img = cv2.cvtColor(self.cylindrical_img, cv2.COLOR_BGR2GRAY)

        self.resp_map = msop_utils.get_corner_resp_map(self.grayscale_img)

        self.hash_map = msop_utils.msop_detect(self.grayscale_img, self.layer_num)


def main(opt: argparse.Namespace):
    stdout_handler = log_utils.get_stream_handler()
    logger = log_utils.get_logger("msop-logger", handlers=[stdout_handler])

    logger.info("Initializing MSOP Image Stitching")

    img_folder = opt.data
    output_name = opt.output
    inlier_thres = opt.inlier_thres

    file_paths = file_utils.get_files(img_folder)
    file_paths.sort()

    imgs = []
    img_frag_list: List[ImageFragment] = []
    for file_path in file_paths:
        _, ext = file_utils.get_extension(file_path)
        if ext not in EXTS:
            continue

        logger.info("found {}".format(file_path))

        imgs.append(cv2.imread(file_path))
        img_frag_list.append(ImageFragment(file_path, focal_length=opt.fc))

    main_frag = img_frag_list[0]
    for idx in range(1, len(img_frag_list)):

        frag = img_frag_list[idx]

        logger.info("processing image {} / {}: {}".format(idx + 1, len(img_frag_list), frag.path))

        mv_pairs = msop_utils.match(main_frag.hash_map, frag.hash_map)
        mv_pairs = msop_utils.mv_filter(main_frag.grayscale_img, mv_pairs)
        mv, inliers = msop_utils.ransac(mv_pairs)

        if inliers < len(mv_pairs) * inlier_thres:
            logger.info("inliers {} / {}".format(inliers, len(mv_pairs)))
            continue

        img_a, img_b, mv = msop_utils.reformat_mv(main_frag.cylindrical_img, frag.cylindrical_img, mv)
        merged = msop_utils.blend_imgs(img_a, img_b, mv)

        main_frag.replace(merged)

    logger.info("merging done for all {} images".format(len(img_frag_list)))

    if isinstance(opt.output, str):
        cv2.imwrite(join(img_folder, output_name), main_frag.cylindrical_img)

    cv2.imshow("Merged", main_frag.cylindrical_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
