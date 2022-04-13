import cv2

from utils import msop as msop_utils


if __name__ == '__main__':
    filename = 'samples/0.jpg'
    img = cv2.imread(filename)

    img_flipped = img[:, ::-1]
    img = img[:, :]

    blended = msop_utils.blending(img, img_flipped, (100, 100))

    cv2.imshow("blended", blended)
    cv2.waitKey(0)
