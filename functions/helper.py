import cv2
import numpy as np


# Define conversions in x and y from pixels space to meters
YM_PER_PIX = 30 / 720 # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700 # meters per pixel in x dimension


def rgb2gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def bgr2gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def rgb2hls(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)


def curvature_rad(curve, y_curve_point):
    return ((1 + (2 * curve[0] * y_curve_point + curve[1]) ** 2) ** 1.5) / np.absolute(2 * curve[0])


def pix_to_real(curve):
    real_curve = np.copy(curve)
    real_curve[0] = curve[0] * XM_PER_PIX / (YM_PER_PIX ** 2)
    real_curve[1] = curve[1] * XM_PER_PIX / YM_PER_PIX
    real_curve[2] = curve[2] * XM_PER_PIX
    return real_curve


def poly_fit_two(x, y):
    return np.polyfit(x, y, 2)


def distance_to_center(left_fit, right_fit):
    return ((left_fit[2] + right_fit[2]) / 2 - 640) * XM_PER_PIX


def extract_xs_and_ys(nonzero_x, nonzero_y, lane_indices):
    return nonzero_x[lane_indices], nonzero_y[lane_indices]


def nonzero_x_and_y(img):
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzero_y, nonzero_x = np.array(nonzero[0]), np.array(nonzero[1])
    return nonzero_x, nonzero_y


def left_and_right_poly_fit(left_y, left_x, right_y, right_x):
    # Fit a second order polynomial to each
    return poly_fit_two(left_y, left_x), poly_fit_two(right_y, right_x)


def parallelization_check(left_fit, right_fit, error_margin):
    return not (abs(left_fit[0] - right_fit[0]) > error_margin[0] or abs(left_fit[1] - right_fit[1]) > error_margin[1])


def averaging_fit(found_lanes, look_back):
    look_back = min([look_back, len(found_lanes)])
    last_few_fits = found_lanes[-look_back:]
    left_fits = [fit[0] for fit in last_few_fits]
    right_fits = [fit[1] for fit in last_few_fits]
    left_fit = sum(left_fits) / look_back
    right_fit = sum(right_fits) / look_back
    return left_fit, right_fit
