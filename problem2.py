import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to load the images and extract the keypoints and descriptors
def load(image):
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)    # Read the image and convert it to RGB
    img = cv2.resize(img, (int(img.shape[1] * 0.2),int(img.shape[0] * 0.2)), interpolation = cv2.INTER_AREA)    # Resize the image
    kp, des = cv2.SIFT_create().detectAndCompute(img, None)   # Extract the keypoints and descriptors
    return img, kp, des

# Function to match the keypoints and find the homography
def matching(img1, img2, kp1, kp2, des1, des2):
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2) # Match keypoints using the brute-force matcher
    good = [m for m,n in matches if m.distance < 0.75*n.distance]   # Select the good matches using the ratio test
    draw_params = dict(matchColor=(0,255,0), singlePointColor=None, flags=2)    # Draw the matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)      
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)    # Extract the matched keypoints 
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)    
    H = findHomography(dst_pts, src_pts)
    return img_matches, H

# Function to find homography using RANSAC algorithm
def findHomography(src_pts, dst_pts, num_iter=100, num_pts=4, threshold=5):
    best_H, best_count = None, 0    # Initialize the best homography and the inlier count
    src_pts, dst_pts = np.hstack((src_pts.reshape(-1,2), np.ones((len(src_pts), 1)))), np.hstack((dst_pts.reshape(-1,2), np.ones((len(dst_pts), 1))))   # Convert the input points to homogeneous coordinates
    
    # Perform RANSAC iterations
    for i in range(num_iter):
        idx = np.random.choice(len(src_pts), num_pts, replace=False)    # Randomly sample num_pts points
        src_sample, dst_sample = src_pts[idx, :], dst_pts[idx, :]
        
        # Construct the matrix A
        A = np.asarray([[x, y, 1, 0, 0, 0, -u*x, -u*y, -u] for (x, y, _), (u, v, _) in zip(src_sample, dst_sample)] + [[0, 0, 0, x, y, 1, -v*x, -v*y, -v] for (x, y, _), (u, v, _) in zip(src_sample, dst_sample)])
        _, _, V = np.linalg.svd(A)  # Perform SVD on matrix A
        H = V[-1, :].reshape(3, 3)  # Extract the homography from the last row of V
        count = sum(np.linalg.norm(np.dot(H, src_pts[j]) / np.dot(H, src_pts[j])[2] - dst_pts[j]) < threshold for j in range(len(src_pts)))    # Count the number of inliers for the current homography
        if count > best_count: best_H, best_count = H, count   # Update the best homography and the inlier count
    return best_H

# Function to trim the black borders of the stitched image    
def trim(frame):
    while not np.any(frame[0]): frame = frame[1:]
    while not np.any(frame[-1]): frame = frame[:-2]
    while not np.any(frame[:,0]): frame = frame[:,1:]
    while not np.any(frame[:,-1]): frame = frame[:,:-2]
    return frame

# Function to stitch the images
def stitch_images(img2, img1, H):
    width = img2.shape[1] + img1.shape[1]
    height = max(img2.shape[0], img1.shape[0])
    stitched_img = cv2.warpPerspective(img2, H, (width,height))  # Warp the image using the homography matrix
    stitched_img[0:img1.shape[0], 0:img1.shape[1]] = img1   
    return stitched_img

img1, kp1, des1 = load('image_1.jpg')
img2, kp2, des2 = load('image_2.jpg')
img3, kp3, des3 = load('image_3.jpg')
img4, kp4, des4 = load('image_4.jpg')

# Draw the matches between the consecutive images
img_matches12, H12 = matching(img1, img2, kp1, kp2, des1, des2)
img_matches23, H23 = matching(img2, img3, kp2, kp3, des2, des3)
img_matches34, H34 = matching(img3, img4, kp3, kp4, des3, des4)

# Display the matched features
plt.figure(figsize=(20,10))
for i, img_match, title in zip(range(1,4), [img_matches12, img_matches23, img_matches34], ['Matches between Image 1 and Image 2', 'Matches between Image 2 and Image 3', 'Matches between Image 3 and Image 4']):
    plt.subplot(1,3,i),plt.imshow(img_match),plt.title(title)
plt.show()

# Stitch the images
stitched_img12 = stitch_images(img2, img1, H12)
stitched_img34 = stitch_images(img4, img3, H34)
stitched_img1234 = stitch_images(trim(stitched_img34), trim(stitched_img12), np.dot(H23, H12))

# Display the final stitched image
plt.imshow(trim(stitched_img1234))
plt.show()