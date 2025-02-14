import numpy as np
import cv2 as cv
import glob

# Chessboard parameters
chessboard_size = (9, 6)  # Internal corners
square_size = 2.5  # Square size in cm

# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# Prepare 3D object points
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Storage for object and image points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load images
images = glob.glob("images for run 1/*.png")

if not images:
    print("No images found! Check the folder path.")
    exit()

# Get image size
img = cv.imread(images[0])
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
image_size = gray.shape[::-1]  # (width, height)

# **Step 1: First Pass - Initial Calibration**
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv.imshow("Initial Detection", img)
        cv.waitKey(200)
    else:       
        print(f"❌ Chessboard not found in {fname}. Please select the 4 outer corners manually.")

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)

# Save the final calibration
np.save("camera_matrix.npy", mtx)
np.save("distortion_coeffs.npy", dist)

print("\nFinal Calibration Complete")
print(f"Final Reprojection Error: {ret}")
print(f"Camera Matrix:\n{mtx}")
print(f"Distortion Coefficients:\n{dist}")
