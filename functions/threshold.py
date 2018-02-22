from .helper import rgb2gray, rgb2hls
import cv2
import numpy as np

K_SIZE = 3
ABS_THRESH = (64, 128)
MAG_K_SIZE = 3
MAG_THRESH = (30, 100)
DIR_K_SIZE = 15
DIR_THRESH = (0.7, 1.3)
SAT_THRESH = (90, 255)
LIT_THRESH = (30, 255)


def thresholding(img, thresh, conversion):
    single_channel = conversion(img)
    binary_output = np.zeros_like(single_channel)
    binary_output[(single_channel >= thresh[0]) & (single_channel <= thresh[1])] = 1
    return binary_output


def absolute_sobel_thresholding(img, orient='x', sobel_kernel=3, abs_thresh=(0, 255)):
    def sobel(image):
        gray = rgb2gray(image)
        x = 1 if orient == 'x' else 0
        return np.abs(cv2.Sobel(gray, cv2.CV_64F, x, 1 - x, ksize=sobel_kernel))

    return thresholding(img, abs_thresh, sobel)


def magnitude_thresholding(img, sobel_kernel=3, mag_thresh=(0, 255)):
    def absolute_sobel(image):
        gray = rgb2gray(image)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel_xy = np.abs(np.sqrt(np.square(sobel_x) + np.square(sobel_y)))
        return np.uint8(255 * abs_sobel_xy / np.max(abs_sobel_xy))

    return thresholding(img, mag_thresh, absolute_sobel)


def direction_thresholding(img, sobel_kernel=3, dir_thresh=(0, np.pi / 2)):
    def grad_dir(image):
        gray = rgb2gray(image)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel_x = np.abs(sobel_x)
        abs_sobel_y = np.abs(sobel_y)
        return np.arctan2(abs_sobel_y, abs_sobel_x)

    return thresholding(img, dir_thresh, grad_dir)


def saturation_thresholding(img, sat_thresh=(0, 255)):
    def saturation_selection(image):
        hls_img = rgb2hls(image)
        return hls_img[:, :, 2]

    return thresholding(img, sat_thresh, saturation_selection)


def lightness_thresholding(img, lit_thresh=(0, 255)):
    def lightness_selection(image):
        hls_img = rgb2hls(image)
        return hls_img[:, :, 1]

    return thresholding(img, lit_thresh, lightness_selection)


def combined_thresholding(img):
    gradx = absolute_sobel_thresholding(img, orient='x', sobel_kernel=K_SIZE, abs_thresh=ABS_THRESH)
    grady = absolute_sobel_thresholding(img, orient='y', sobel_kernel=K_SIZE, abs_thresh=ABS_THRESH)
    mag_binary = magnitude_thresholding(img, sobel_kernel=MAG_K_SIZE, mag_thresh=MAG_THRESH)
    dir_binary = direction_thresholding(img, sobel_kernel=DIR_K_SIZE, dir_thresh=DIR_THRESH)
    sat_binary = saturation_thresholding(img, SAT_THRESH)
    lit_binary = lightness_thresholding(img, LIT_THRESH)
    combined = np.zeros((img.shape[0], img.shape[1]))
    # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1) & (sat_binary == 1))] = 1
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (
            (sat_binary == 1) & (lit_binary == 1))] = 1
    # combined[((gradx == 1) & (grady == 1) & (lit_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | ((
    #             sat_binary == 1) & (lit_binary == 1))] = 1
    return combined
