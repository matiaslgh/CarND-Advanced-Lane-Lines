## Advanced Lane Finding

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

## Overview

The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Run locally

```
conda env create -f environment.yml
conda activate CarND-Advanced-Lane-Lines
jupyter notebook notebook.ipynb
```

## Deliverables

[Writeup](https://github.com/matiaslgh/CarND-Advanced-Lane-Lines/blob/master/WRITEUP.md)

[Jupiter Notebook](https://github.com/matiaslgh/CarND-Advanced-Lane-Lines/blob/master/notebook.ipynb)

[Output Video](https://github.com/matiaslgh/CarND-Advanced-Lane-Lines/blob/master/output.mp4)
