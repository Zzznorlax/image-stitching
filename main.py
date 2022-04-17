import cv2

from utils import image as img_utils
from utils import msop as msop_utils
from utils import visualization as vis_utils


if __name__ == '__main__':

    p1 = "samples/02.jpg"
    p2 = "samples/01.jpg"

    imgs = [cv2.imread(p1), cv2.imread(p2)]
    cy_imgs = []
    resp_maps = []

    hash_maps = []

    # cv2.imshow("t", img_utils.to_cylindrical(imgs[0], 710))
    # cv2.waitKey(0)

    for img in imgs:
        img = img_utils.cylindrical_projection(img, 710)
        cy_imgs.append(img)

        img = img[:, :, 0]

        resp_maps.append(msop_utils.get_corner_resp_map(img))

        hash_map = msop_utils.msop_detect(img, layer_num=1)
        hash_maps.append(hash_map)
        # print(hash_map.patch_count)

    print("Finished building hashmap")
    mv_pairs = msop_utils.match(hash_maps[0], hash_maps[1])
    print("Finished matching")

    print("RANSAC-ing")
    mv = msop_utils.ransac(mv_pairs)

    print(mv)

    # mv = (0, 0)

    result = msop_utils.blend_imgs(cy_imgs[0], cy_imgs[1], mv)

    # cv2.imshow("stitched", result)
    # cv2.waitKey(0)

    import matplotlib.pyplot as plt

    # imgs = [vis_utils.show_resp(cy_imgs[0], resp_maps[0]), vis_utils.show_resp(cy_imgs[1], resp_maps[1])]
    imgs = [
        vis_utils.show_resp(cy_imgs[0], resp_maps[0]),
        vis_utils.show_resp(cy_imgs[1], resp_maps[1]),
        vis_utils.show_matched(cy_imgs[0], cy_imgs[1], mv_pairs)
    ]

    cv2.imshow("matched", vis_utils.show_matched(cy_imgs[0], cy_imgs[1], mv_pairs))
    cv2.waitKey(0)

    # label_list = ['Original', 'keypoint', 'trace', 'det']
    # for idx in range(len(imgs)):
    #     plt.subplot(2, 2, idx + 1)
    #     plt.title(label_list[idx])
    #     plt.imshow(imgs[idx], interpolation='none')
    #     plt.xticks([])
    #     plt.yticks([])

    # plt.show()
