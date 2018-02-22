# Writeup

## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_calibration.jpg "Undistorted Calibration"
[image2]: ./test_images/straight_lines1.jpg "Road Transformed"
[image3]: ./output_images/undistorted_straight_lines.jpg "Undistorted Straight Line"
[image4]: ./output_images/binary_test5.jpg "Binary Example"
[image5]: ./output_images/warp_straight_lines1.jpg "Warp Example"
[image6]: ./output_images/fit_test5.jpg "Fit Visual"
[image7]: ./output_images/test2.jpg "Output"
[video1]: ./project_video_output.mp4 "Output Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image

The code for this step is contained in lines 9 through 53 of the file called `functions/camera_calibration.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpts` and `imgpts` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

With the functions created above in `functions/camera_calibration.py`, I'd apply calibration based on the image size (`calibrate`) and then undistort the image with the matrix and distortion coefficients (`undistort`). The end result is like this:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result

I used a combination of color and gradient thresholds to generate a binary image (thresholding functions in `threshold.py`). Comparing to the examples in the class, I raised the absolute Sobel threshold so it picks up less wiggy lines between the lane lines. Also, saturation thresholding is used to enhance the lane lines as they are better at picking up the consistently saturated colors. Then I dialed it down a little bit with lightness threshold since shadows look saturated but dark.

Here's an example of my output for this step.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 8 through 9 in the file `perspective_transform.py` (functions/perspective_transform.py).  The `warp_image()` function takes as inputs an image (`img`), as well as perspective transform matrix (`matrix`). The matrix is generated using `perspective_transform_matrix()` function, which takes in source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points by hand based on rough estimates in one of straight line image:

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 574, 461      | 320, 0        |
| 185, 720      | 320, 720      |
| 1122, 720     | 960, 720      |
| 706, 461      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial

Then I start with the histgram, find the two peaks (left and right), and use them as the bases of Sliding Window. The search is done with 100 pixels margin, 50 minimum pixel count per window and 9 total windows.

When both lanes are found in the previous search, I'll just do a margin search with again 100 pixels. If any is missing, the Sliding Window search will be done

I fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center

I did this in lines 22 through 31 and 38 through 39 in my code in `helper.py` and lines 17 through 29 in the code block of `process_image` method in `advanced_lane_lines.ipynb`.

First, I converted the curve I found from previous steps to real world scale with `pix_to_real` method. Then, I used that curve to calculate the curvature at the bottom of the images. Inside that calculation, the formula given in "Measuring Curvature" was used.

Based on the assumption these two curves are parallel, I used the difference of two curves' interceptions to convert to real world distance in `distance_to_center` method.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly

I implemented this step in lines 24 through 26 in my code in `advanced_lane_lines.ipynb` in the function `process_image()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

Here's a [link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust

1. The thresholding is not performing as good in more complicated scenarios. It picks up shadows incorrectly despite the effort to reduce that. The technique I used was to have a lightness filter since the shadows are dark. It helped a little with the problem.

2. The Sliding Window Search and Margin Search performed reasonaly well when no shadow or other objects present nearby. I put in a parallelization check after it identified the lane lines and threw away the results that failed the check. This made sure almost every frame not paint an area larger than the lane area itself. This would not work without the averaging of curves, either. It's possible to add more sanity checks so the result is more filtered and accurate.

3. With a pipeline processing approach like this, it's a bit of challenge to turn it into a maintainable software system. This would leave it vulnerable to software bugs when further developing it. I tried to group the functions into files to make some sense out of the scattered functionalities. More design and refactoring will be needed to improve the code quality.
