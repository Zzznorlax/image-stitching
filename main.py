import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import image as img_utils
from utils import msop as msop_utils
from utils import visualization as vis_utils


if __name__ == '__main__':

    p1 = "samples/01.jpg"
    p2 = "samples/02.jpg"

    imgs = [cv2.imread(p1), cv2.imread(p2)]
    cy_imgs = []
    resp_maps = []

    hash_maps = []

    for img in imgs:
        img = img_utils.cylindrical_projection(img, 710)
        cy_imgs.append(img)

        img = img[:, :, 0]

        resp_maps.append(msop_utils.get_corner_resp_map(img))

        hash_map = msop_utils.msop_detect(img, layer_num=1)
        hash_maps.append(hash_map)

    print("Finished building hashmap")
    mv_pairs = msop_utils.match(hash_maps[0], hash_maps[1])
    print("Finished matching")

    mv_pairs = msop_utils.mv_filter(cy_imgs[0], mv_pairs)

    print("RANSAC-ing")
    mv = msop_utils.ransac(mv_pairs)

    m = resp_maps[0]
    print(mv)

    for i in range(len(cy_imgs)):
        cy_imgs[i] = vis_utils.show_resp(cy_imgs[i], resp_maps[i])

    result = msop_utils.blend_imgs(cy_imgs[0], cy_imgs[1], mv)

    # cv2.imshow("blended", result)

    # cv2.imshow("matched", vis_utils.show_matched(cy_imgs[0], cy_imgs[1], mv_pairs))
    # cv2.waitKey(0)

    imgs = [vis_utils.show_matched(cy_imgs[0], cy_imgs[1], mv_pairs), result]

    label_list = ['Matched features', 'Merged']

    loc_list = [1, 3]
    for idx in range(len(imgs)):
        plt.subplot(2, 2, loc_list[idx])
        plt.title(label_list[idx])
        plt.imshow(imgs[idx], interpolation='none')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()
