import cv2


def perspective_transform_matrix(src, dst):
    return cv2.getPerspectiveTransform(src, dst)


def warp_image(matrix, img):
    return cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)