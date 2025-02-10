import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Mouse callback function to manually select corners
def select_corners(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(param['corners']) < 4:
            param['corners'].append((x, y))
            print(f"Corner {len(param['corners'])} selected at ({x}, {y})")
            if len(param['corners']) == 4:
                print("All 4 corners selected.")

images = glob.glob("images/*.png")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        print("Could not find chessboard in " + fname)
        # print("Please manually select the 4 corners of the chessboard.")

        # # Initialize list to store manually selected corners
        # manual_corners = []

        # # Create a named window and set the mouse callback
        # cv.namedWindow('img')
        # cv.setMouseCallback('img', select_corners, {'corners': manual_corners})

        # while len(manual_corners) < 4:
        #     cv.imshow('img', img)
        #     cv.waitKey(1)

        # # Once 4 corners are selected, compute the homography and find the inner corners
        # manual_corners = np.array(manual_corners, dtype=np.float32)

        # # Define the destination points for the homography (assuming the chessboard is 9x6)
        # dst_points = np.array([[0, 0], [8, 0], [8, 5], [0, 5]], dtype=np.float32)

        # # Compute the homography matrix
        # H, _ = cv.findHomography(manual_corners, dst_points)

        # # Use the homography to find the inner corners
        # h, w = gray.shape
        # inner_corners = cv.perspectiveTransform(np.array([[[0, 0], [8, 0], [8, 5], [0, 5]]], dtype=np.float32), H)
        # inner_corners = inner_corners.reshape(-1, 2)

        # # Append the object points and image points
        # objpoints.append(objp)
        # imgpoints.append(inner_corners)

        # # Draw the manually selected corners
        # for corner in manual_corners:
        #     cv.circle(img, tuple(corner.astype(int)), 5, (0, 255, 0), -1)

        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()