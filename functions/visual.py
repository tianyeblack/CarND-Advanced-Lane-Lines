import numpy as np
import cv2

PLOT_Y = np.array(list(range(0, 721)))
R, G, B, Y = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)


def left_and_right_plot_x(left_fit, right_fit):
    return np.polyval(left_fit, PLOT_Y), np.polyval(right_fit, PLOT_Y)


def left_and_right_pts(plot_left_x, plot_right_x):
    pts_left = np.array([np.transpose(np.vstack((plot_left_x, PLOT_Y)))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack((plot_right_x, PLOT_Y))))])
    return pts_left, pts_right


def visualize_fit(left_fit, right_fit, fit_visual):
    plot_left_x, plot_right_x = left_and_right_plot_x(left_fit, right_fit)
    pts_left, pts_right = left_and_right_pts(plot_left_x, plot_right_x)
    cv2.polylines(fit_visual, np.int_([pts_left]), False, Y)
    cv2.polylines(fit_visual, np.int_([pts_right]), False, Y)
    return fit_visual


def visualize_margin(left_fit, right_fit, fit_visual, margin=100):
    margin_visualization = np.uint8(np.zeros_like(fit_visual))
    plot_left_x, plot_right_x = left_and_right_plot_x(left_fit, right_fit)
    pts_left_left, pts_left_right = left_and_right_pts(plot_left_x - margin, plot_left_x + margin)
    pts_right_left, pts_right_right = left_and_right_pts(plot_right_x - margin, plot_right_x + margin)
    cv2.fillPoly(margin_visualization, np.int_([np.hstack((pts_left_left, pts_left_right))]), G)
    cv2.fillPoly(margin_visualization, np.int_([np.hstack((pts_right_left, pts_right_right))]), G)
    return cv2.addWeighted(fit_visual, 1, margin_visualization, 0.3, 0)


def visualize_lane_area(left_fit, right_fit, img):
    lane_area_visualization = np.uint8(np.zeros_like(img))
    plot_left_x, plot_right_x = left_and_right_plot_x(left_fit, right_fit)
    pts_left, pts_right = left_and_right_pts(plot_left_x, plot_right_x)
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(lane_area_visualization, np.int_([pts]), G)
    return lane_area_visualization


def visualize_texts(img, curvature, dist_to_ctr):
    cv2.putText(img, "Radius of Curvature = %sm" % curvature, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, G)
    cv2.putText(img, "Vehicle is %sm %s of center" % (dist_to_ctr, ("left" if dist_to_ctr < 0 else "right")), (50, 100),
                cv2.FONT_HERSHEY_PLAIN, 2.0, G)
    return img
