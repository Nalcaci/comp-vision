import numpy as np
import cv2 as cv
import glob

# Termination criteria for refining detected corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real-world space)
square_size = 25  # Square size in mm
objp = np.zeros((7 * 10, 3), np.float32)
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * square_size  # Scale by square size

# Arrays to store object points and image points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load all images in the 'images' folder (change if needed)
images = glob.glob("images/*.png")  # Make sure your images are in this folder

if not images:
    print("No images found! Check your path or image format.")
    exit()

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (10, 7), None)

    # If found, add object and image points
    if ret:
        objpoints.append(objp)

        # Refine corner locations
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (10, 7), corners2, ret)
        cv.imshow("Chessboard Detection", img)
        cv.waitKey(500)

cv.destroyAllWindows()

