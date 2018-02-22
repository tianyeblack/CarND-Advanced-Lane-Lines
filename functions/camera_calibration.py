import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def calibration_preparation(nx, ny):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = sorted(glob.glob('./camera_cal/calibration*.jpg'), key=os.path.basename)
    n_images = len(images)

    # Step through the list and search for chessboard corners
    fig = plt.figure()
    n_plot_rows = 5
    n_plot_cols = 4
    for n, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        fig.add_subplot(n_plot_rows, n_plot_cols, n + 1)
        plt.imshow(img)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    return objpoints, imgpoints


def calibrate(objpoints, imgpoints, img_size):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist


def undistort(mtx, dist, img):
    return cv2.undistort(img, mtx, dist, None, mtx)
