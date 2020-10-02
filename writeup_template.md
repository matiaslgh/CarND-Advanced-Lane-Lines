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
