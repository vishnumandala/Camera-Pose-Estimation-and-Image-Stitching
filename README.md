# PROJECT 2
## ENPM673 - Perception for Autonomous Robots

## Dependencies
1. python 3.11 (any version above 3 should work)
2. Python running IDE (I used VS Code)

## Libraries
1. OpenCV
2. NumPy
3. Matplotlib
4. Math

## Contents
1. project2.avi
2. image_1.jpg
3. image_2.jpg
4. image_3.jpg
5. image_4.jpg
6. K_matrix.xlsx
7. vishnum_proj2.pdf
8. README.md
9. problem1_output.png
10. problem2_matches.png
11. problem2_panorama.png

## Installation Instructions
1. Download the zip file and extract it
2. Install python and the required dependencies: pip install opencv-python numpy matplotlib math

## Problem 1 - Camera Pose Estimation
This code finds the rotation and translation between camera and a coordinate frame whose origin is located at one of the corners of a paper

### Features
1. Extracts the coordinates of the corners of the paper using Hough Transformation
2. Computes Homography between real world coordinates and pixel coordinates of the corners
3. Decomposes the homography matrix to get rotation and translation
4. Displays the Roll, Pitch, Yaw and translations in x, y, z directions on a plot

### Usage
1. Place the video file in the same directory as the code file
2. Set the filename of the video in the cv2.VideoCapture() function call in line 148 of the code
3. Set the paper height and paper width in line 6 of the code
4. Set the intrinsic matrix of the camera in line 135 of the code
5. Run the code: problem1.py

### Example Output

Rotation:
 [[-0.47551727  0.14923454 -0.86695581]
 [ 0.87931474  0.11003532 -0.46335496]
 [-0.02624719  0.98266031  0.18354783]]
Translation:
 [-1.0023014   5.66139708  2.78463452]
 
 
Rotation:
 [[ 0.55341795  0.15025172 -0.81923927]
 [-0.8328733   0.10823348 -0.54277765]
 [ 0.00711584  0.98270542  0.185039  ]]
Translation:
 [ 4.46735501 -3.45086053  2.95554201]

## Problem 2 - Image Stitching
This code stictches given images and generates a Panoramic Image

### Features
1. Extracts features from images using SIFT
2. Uses BFMatcher to match features between images
3. Computes homography for images using RANSAC
4. Combines the images using the homography matrices

### Usage
1. Place the data files in the same directory as the code file
2. Set the filenames of the images in the load function call in lines 57-60 of the code
3. Run the code: problem2.py
