**Advanced Lane Finding Project**

The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"
[chessboard_points]: ./output_images/chessboard_points.jpg "Chessboard points"
[undistorted_chessboard]: ./output_images/undistorted_chessboard.jpg "Undistorted Chessboard"
[undistorted_lane]: ./output_images/undistorted_lane.jpg "Undistorted Lane"
[sobel_x]: ./output_images/sobel_x.jpg "Sobel X"
[saturation_channel]: ./output_images/saturation_channel.jpg "Binary Saturation Channel"
[luma_channel]: ./output_images/luma_channel.jpg "Binary Luma Channel"
[get_binary]: ./output_images/get_binary.jpg "Get Binary Function Result"
[warped_straight_lines]: ./output_images/warped_straight_lines.jpg "Warped Straight Lines"
[warped_curved_lines]: ./output_images/warped_curved_lines.jpg "Warped Curved Lines"
[detected_lines]: ./output_images/detected_lines.jpg "Detected Lines"
[detected_lines_from_polynomial]: ./output_images/detected_lines_from_polynomial.jpg "Detected Lines From Polynomial"
[area_between_lines]: ./output_images/area_between_lines.jpg "Area between lines"

**Important note**

The sections' number in this writeup match with the ones in the jupyter notebook. You can check the code in those sections if you consider that the one shown here is not enough

### 1. Camera Calibration

From a set of chessboard pictures taken with the camera used for the project, we try to match the object points (the corners we know the chessboard has) with the image points (the actual corners in the image)

```python
for img in calibration_img_list:
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_img, (CORNERS_X, CORNERS_Y), None)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        corners_img = cv2.drawChessboardCorners(img, (CORNERS_X, CORNERS_Y), corners, ret)
        corners_img_list.append(corners_img)
    else:
        print('Not found')
```

![alt text][chessboard_points]

### 2. Undistortion

In order to avoid negative effects in our final calculations, we need to remove the camera distortion. To do that, we'll use the camera matrix and the distortion coefficients got from the camera calibration

```python
cv2.undistort(img, mtx, dist, None, mtx)
```

![alt text][undistorted_chessboard]
![alt text][undistorted_lane]

### 3. Get binary image result

To find the lane lines we'll need a binary image that ideally contains only the lane lines and nothing else. That ideal scenario is not possible but there are several techniques that allows us to get pretty good results though.

I tried by using the sobel function and filtering the results by a range based on gradient direction and gradient magnitude, but I got better results by just using sobel with a kernel that only compares left/right changes

I also tried by using the different channels of different color spaces, and I got the best results by using the saturation channel of the HSL color space and the luma channel of the YUV color space. They provide good results in different scenarios, that's why it's a good idea to combine them.

![alt text][sobel_x]
![alt text][saturation_channel]
![alt text][luma_channel]

Finally I merged those results to build the function `get_binary`

```python
def get_binary(image):
    bin_sat_img = sat_channel_threshold(image, SAT_CHANNEL_THRESHOLD)
    bin_luma_img = y_channel_threshold(image, LUMA_CHANNEL_THRESHOLD)
    sobelx_img = abs_sobelx_thresh(image, SOBEL_X_THRESHOLD, SOBEL_X_KERNEL)

    binary_merge = np.zeros_like(sobelx_img)
    binary_merge[((bin_sat_img == 1) & (bin_luma_img == 1)) | (sobelx_img == 1)] = 1

    return binary_merge
```

In the image below we can see how the combination of these binary results provides different information of the same image. Red is luma and saturation combined (when both are 1 in the same pixel), and green is sobel X. I applied an OR between those 2 `(luma and saturation) or sobel_x`

![alt text][get_binary]

### 4. Apply birds-eye view transformation

The idea is to identify the lane lines from a birds-eye view as well as the curvature of the curve. To do that we have to create a transformation function with `M = cv2.getPerspectiveTransform(src, dst)` and the use that apply it with `cv2.warpPerspective(undistorted_img, M, img_size, flags=cv2.INTER_LINEAR)`

So, the red line represents the `src` points and the green the `dst`

![alt text][warped_straight_lines]
![alt text][warped_curved_lines]

### 5. Detect lane lines

- Get the binary version of the warped image
- Create histogram
- Use histogram to find the mean X value where are most of the pixels for both left and right lane lines
  - Sometimes, for curves, setting the first window with center in the mean doesn't detect any pixel. If so, use a wider window as base.
- With those initial positions, draw small windows to detect the lane pixels in those windows
- Recenter every window if the previous one had enough amount of pixels
- Return the indices of the pixels that are lane lines
- Use lane pixels to get a polynomial function that fits with those values

![alt text][detected_lines]

That algorithm is to find lane lines pixels from scratch, but since these lines won't change too much their position between a frame and the next one, we can create a function that receives the polynomial coefficients from previous frame to try to search in a specific area around that polynomial. So, instead of using windows, it would look like:

![alt text][detected_lines_from_polynomial]

### 6. Messure curvature

Calculate radius based on this formula: https://www.intmath.com/applications-differentiation/8-radius-curvature.php

```python
Y_METERS_PER_PIXEL = 30 / 720

def measure_curvature(coefficients, ploty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    A, B, C = coefficients
    return ((1 + (2 * A * y_eval * Y_METERS_PER_PIXEL + B)**2)**1.5) / np.absolute(2 * A)
```

### 7. Messure distance from center

```python
X_METERS_PER_PIXEL = 3.7 / 700

def distance_from_center(x_length, left_fit, right_fit, ploty):
    car_position= x_length / 2

    y_eval=np.max(ploty)

    left_lane_bottom = (left_fit[0] * y_eval)**2 + left_fit[0] * y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0] * y_eval)**2 + right_fit[0] * y_eval + right_fit[2]

    center_position = (left_lane_bottom + right_lane_bottom)/2

    return (car_position - center_position) * X_METERS_PER_PIXEL
```

### 8. Draw area between lane lines

Use the found polynomial to draw the area between the lane lines

![alt text][area_between_lines]

### 9. Define the final pipeline

- undistort
- get binary
- get birds-eye view
- are previous polynomial coefficients?
  - NO! --> search lane lines pixels by using histogram
  - YES! --> search in that specific area. If fails, use
- get new polynomial coefficients
  - if fails, use previous values and in next iterations earch from pixels from histogram
  - if works, set previous coefficients
- use polynomial coefficients to draw are between lines
- calculate and show radius
- calculate and show distance from center

### 10. Apply all to a video!

Use the pipeline! You can downlowd my output.mp4 [here](https://github.com/matiaslgh/CarND-Advanced-Lane-Lines/blob/master/output.mp4)

### Thoughts

_Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?_

- Still struggling with shadows
- When the lane is not completely plane, the trasformation fails really bad. This does not work with the challenge videos
- I feel that the currect hyperparameter configuration works quite good with the project video but it won't work ok in tougher scenarios
