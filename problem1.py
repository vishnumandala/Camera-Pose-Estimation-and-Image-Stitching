import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

paper_width, paper_height = 21.6, 27.9  # Define the dimensions of the paper in cm
real_world_corners = np.array([[0, 0], [0, paper_height], [paper_width, paper_height], [paper_width, 0]])   # Define the coordinates of the corners of the paper in the real world coordinate system

# Define the function to extract the corners of the paper
def extract_paper_corners(img):
    gray = cv2.cvtColor(cv2.bitwise_and(img, img, mask=cv2.inRange(img, (200, 120, 100), (255, 255, 255))), cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale and extract the white paper using color masking
    gray = cv2.bilateralFilter(gray, 9, 75, 75)     # Apply bilateral filter to the grayscale image
    blurred = cv2.GaussianBlur(gray, (3, 3), 1)     # Apply Gaussian blur to the grayscale image
    closed = cv2.Canny(blurred, 750, 770)           # Apply Canny edge detection to the blurred image

    # Define the range of theta and rho
    height, width = closed.shape
    theta_range = np.deg2rad(np.arange(-90, 90, 1))
    rho_max = int(math.ceil(math.sqrt(height**2 + width**2)))
    rho_range = np.arange(-rho_max, rho_max + 1, 1)

    # Create accumulator array and accumulate votes
    accumulator = np.zeros((len(rho_range), len(theta_range)))
    for y, x in np.argwhere(closed):
        rho_vals = np.round(x * np.cos(theta_range) + y * np.sin(theta_range)).astype(int) + rho_max
        accumulator[rho_vals, np.arange(len(theta_range))] += 1
        
    peaks = np.argwhere(accumulator > 120)  # Select the peaks with votes greater than 120 (threshold) and store them in peaks

    # Extract the (rho, theta) values corresponding to the peaks
    lines = []
    max_votes = [0] * 4
    final_lines = [None] * 4
    # Find the lines that are similar to the lines in lines
    for rho_idx, theta_idx in peaks:
        rho, theta = rho_range[rho_idx], theta_range[theta_idx]
        similar = False
        for i, (prev_rho, prev_theta) in enumerate(lines):
            if abs(rho - prev_rho) < 200 and abs(theta - prev_theta) < np.deg2rad(120):
                similar = True
                break
        if not similar:
            lines.append((rho, theta))
            votes = accumulator[rho_idx, theta_idx]
            if votes > min(max_votes):
                idx = max_votes.index(min(max_votes))
                max_votes[idx] = votes
                final_lines[idx] = (rho, theta)

    intersections = []
    for i, (rho, theta) in enumerate(lines):
        # Convert from polar coordinates to Cartesian coordinates and plot the line
        x0, y0 = rho * np.cos(theta), rho * np.sin(theta)
        x1, y1 = np.array([x0, y0]) + 2000 * np.array([-np.sin(theta), np.cos(theta)])
        x2, y2 = np.array([x0, y0]) - 2000 * np.array([-np.sin(theta), np.cos(theta)])
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        #Find the intersection of the current line with all the other lines
        for rho1, theta1 in lines[i+1:]:
            a = np.array([[np.cos(theta), np.sin(theta)], [np.cos(theta1), np.sin(theta1)]])
            b = np.array([rho, rho1])
            try:
                intersection = tuple(map(int, np.linalg.solve(a, b)))
                intersections.append(intersection)
            except np.linalg.LinAlgError:
                pass
            
    corners = np.array(intersections)   # Convert the list of intersections to a numpy array

    return corners

# Define the function to compute the homography matrix
def compute_homography(real_world_points, image_points):
    # Build the A matrix
    A = []
    for i in range(4):
        x, y = real_world_points[i]
        u, v = image_points[i]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)

    # Solve for the homography matrix using linear least squares
    _, _, V = np.linalg.svd(A)  # Compute the SVD of A
    H = (V[-1,:] / V[-1,-1]).reshape((3,3)) # Extract the last column of V and reshape it to a 3x3 matrix

    return H

# Define the function to decompose the homography matrix
def decompose_homography(H):
    K = np.array([[1.38e+03, 0, 9.46e+02], [0, 1.38e+03, 5.27e+02], [0, 0, 1]])  # Camera matrix
    H = np.linalg.inv(K) @ H    # Compute the homography matrix with respect to the camera frame
    R = H[:, :3]    # Extract the first three columns of the homography matrix

    # Normalize the first three columns to obtain the rotation matrix
    U, _, Vt = np.linalg.svd(R) 
    R = np.dot(U, Vt)   

    t = H[:, 2] / np.linalg.norm(H[:, :2])      # Extract the last column of the homography matrix and normalize it
    t /= H[2, 2]                                # Divide the last column by the last element of the last row of the homography matrix
    t = -np.dot(R, t)                           # Compute the translation vector by multiplying the rotation matrix with the last column of the homography matrix
    return R, t

cap = cv2.VideoCapture('project2.avi')
roll_list, pitch_list, yaw_list, x_list, y_list, z_list = [], [], [], [], [], []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    corners = extract_paper_corners(frame)
    for corner in corners:
        cv2.circle(frame, tuple(corner), 5, (0, 0, 255), -1)    # Draw the corners on the image

    H = compute_homography(real_world_corners, corners)
    R, t = decompose_homography(H)
    print("\nRotation:\n", R, "\nTranslation:\n", t, '\n')  # Print the rotation and translation matrices

    roll, pitch, yaw = [math.atan2(R[i, j], R[i, k]) for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]]  # Compute the roll, pitch, and yaw angles from the rotation matrix
    tx, ty, tz = t  # Extract the translation vector components

    roll_list.append(roll)
    pitch_list.append(pitch)
    yaw_list.append(yaw)
    x_list.append(tx)
    y_list.append(ty)
    z_list.append(tz)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

# Plot the roll, pitch, and yaw data in the first subplot
ax1.plot(roll_list, label='Roll')
ax1.plot(pitch_list, label='Pitch')
ax1.plot(yaw_list, label='Yaw')
ax1.legend()
ax1.set_ylabel('Angle (rad)')

# Plot the x, y, and z data in the second subplot
ax2.plot(x_list, label='X')
ax2.plot(y_list, label='Y')
ax2.plot(z_list, label='Z')
ax2.legend()
ax2.set_ylabel('Translation (cm)')
ax2.set_xlabel('Frame')

# Show the plot
plt.show()