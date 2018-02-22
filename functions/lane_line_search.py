from .helper import extract_xs_and_ys, left_and_right_poly_fit, nonzero_x_and_y, poly_fit_two
import numpy as np
import cv2

R, G, B, Y = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)


def sliding_window(img, no_of_win, win_margin, min_pixels, color):
    # Create an output image to draw on and  visualize the result
    out_img = np.uint8(np.dstack((img, img, img)) * 255)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)
    # Find the peak of the histogram, which will be the starting point for the lane line
    x_base = np.argmax(histogram)
    # Set height of windows
    window_height = np.int(img.shape[0] / no_of_win)

    nonzero_x, nonzero_y = nonzero_x_and_y(img)
    x_current = x_base
    lane_indices = []

    # Step through the windows one by one
    for win in range(no_of_win):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (win + 1) * window_height
        win_y_high = img.shape[0] - win * window_height
        win_x_low = x_current - win_margin
        win_x_high = x_current + win_margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), G, 2)
        # Identify the nonzero pixels in x and y within the window
        good_indices = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                        (nonzero_x >= win_x_low) & (nonzero_x < win_x_high)).nonzero()[0]
        # Append these indices to the lists
        lane_indices.append(good_indices)
        # If you found > min_pixels pixels, recenter next window on their mean position
        if len(good_indices) > min_pixels:
            x_current = np.int(np.mean(nonzero_x[good_indices]))

    # Concatenate the arrays of indices
    lane_indices = np.concatenate(lane_indices)

    # Extract left and right line pixel positions
    x, y = extract_xs_and_ys(nonzero_x, nonzero_y, lane_indices)

    #     print("sliding_window(%s)" % out_img[242, 620])
    out_img[y, x] = color
    return x, y, out_img


def sliding_window_2_lanes(img, no_of_win=9, win_margin=100, min_pixels=50):
    midpoint = int(img.shape[1] / 2)
    left_x, left_y, left_half = sliding_window(img[:, :midpoint], no_of_win, win_margin, min_pixels, R)
    right_x, right_y, right_half = sliding_window(img[:, midpoint:], no_of_win, win_margin, min_pixels, B)
    right_x += midpoint

    left_fit, right_fit = left_and_right_poly_fit(left_y, left_x, right_y, right_x)
    fit_visual = np.hstack((left_half, right_half))
    return left_fit, right_fit, fit_visual


def margin_search(img, last_fit, margin, color):
    # Create an output image to draw on and  visualize the result
    out_img = np.uint8(np.dstack((img, img, img)) * 255)
    nonzero_x, nonzero_y = nonzero_x_and_y(img)
    fit_x = np.polyval(last_fit, nonzero_y)
    lane_indices = ((nonzero_x > fit_x - margin) & (nonzero_x < fit_x + margin))
    x, y = extract_xs_and_ys(nonzero_x, nonzero_y, lane_indices)
    out_img[y, x] = color
    return x, y, out_img


def margin_search_2_lanes(last_left_fit, last_right_fit, img, margin=100):
    midpoint = int(img.shape[1] / 2)
    left_x, left_y, left_half = margin_search(img[:, :midpoint], last_left_fit, margin, R)
    right_x, right_y, right_half = margin_search(img[:, midpoint:], last_right_fit, margin, B)
    right_x += midpoint

    left_fit = right_fit = None
    if len(left_x) > 0 and len(left_y) > 0:
        left_fit = poly_fit_two(left_y, left_x)
    if len(right_x) > 0 and len(right_y) > 0:
        right_fit = poly_fit_two(right_y, right_x)
    fit_visual = np.hstack((left_half, right_half))
    return left_fit, right_fit, fit_visual


def lane_line_search(last_left_fit, last_right_fit, img):
    left_fit = right_fit = fit_visual = None
    if last_left_fit is not None and last_right_fit is not None:
        left_fit, right_fit, fit_visual = margin_search_2_lanes(last_left_fit, last_right_fit, img)

    if left_fit is None or right_fit is None or last_left_fit is None or last_right_fit is None:
        left_fit, right_fit, fit_visual = sliding_window_2_lanes(img)
    #     print(left_fit, right_fit, parallelization_check(left_fit, right_fit, [0.0001, 0.07]))
    return left_fit, right_fit, fit_visual
