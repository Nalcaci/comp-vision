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
images = glob.glob("images for run 3/*.png")

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

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        # objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        # imgpoints.append(corners2)

        img_height, img_width = img.shape[:2]
        cameraMatrix = mtx

        ret2, rvec, tvec = cv.solvePnP(objp, corners2, cameraMatrix, dist)
        if ret2 == True:
            board_center = np.mean(objp, axis=0).reshape(1, 3)

            axis_length = square_size * 2  # e.g., length equal to three squares
            # Axes: X (red), Y (green) and Z (blue).
            # (Here, the Z axis is drawn in the negative direction so that it “rises” from the board.)
            axes_points = np.float32([
                [0, 0, 0],                        # origin (board center)
                [axis_length, 0, 0],               # X axis
                [0, axis_length, 0],               # Y axis
                [0, 0, -axis_length]               # Z axis (pointing upward from board)
            ])
            # Place the axes at the board center.
            axes_world = board_center + axes_points
            axes_img, _ = cv.projectPoints(axes_world, rvec, tvec, cameraMatrix, dist)
            axes_img = axes_img.reshape(-1, 2).astype(int)
            
            # Draw the axes lines on the image.
            origin_pt = tuple(axes_img[0])
            cv.line(img, origin_pt, tuple(axes_img[1]), (0, 0, 255), 3)  # X axis in red
            cv.line(img, origin_pt, tuple(axes_img[2]), (0, 255, 0), 3)  # Y axis in green
            cv.line(img, origin_pt, tuple(axes_img[3]), (255, 0, 0), 3)  # Z axis in blue
            
            # -----------------------------
            # DEFINE THE CUBE
            # -----------------------------
            # We define a cube whose bottom face is centered at the board center (on the chessboard plane, z=0)
            # and whose top face is at z = -cube_size (extending upward).
            cube_size = axis_length  # set cube side length; adjust as desired
            half = cube_size / 2.0
            
            # Define bottom face vertices (relative to board_center)
            # Instead of computing board_center and a centered square,
            # Define cube size to cover 2 squares by 2 squares on the chessboard.
            cube_size = 2 * square_size  # each side of the cube's base spans 2 squares.
            half = cube_size / 2.0
            
            # Define the cube to cover a 2x2 block of chessboard squares.
            # Here we choose the block with chessboard object coordinates:
            # bottom-left (3,2), bottom-right (5,2), top-right (5,4), and top-left (3,4)
            bottom_face = np.float32([
                [3 * square_size, 2 * square_size, 0],
                [5 * square_size, 2 * square_size, 0],
                [5 * square_size, 4 * square_size, 0],
                [3 * square_size, 4 * square_size, 0]
            ])

            # For a cube, set its height equal to 2 squares.
            cube_height = 2 * square_size
            # Define the top face by shifting the bottom face upward (using -cube_height)
            top_face = bottom_face.copy()
            top_face[:, 2] = -cube_height  # negative z to "raise" the cube

            # Combine the bottom and top face vertices.
            cube_world = np.concatenate((bottom_face, top_face), axis=0)

            # Project the cube's 3D points to 2D image points.
            cube_img, _ = cv.projectPoints(cube_world, rvec, tvec, cameraMatrix, dist)
            cube_img = cube_img.reshape(-1, 2).astype(int)

            # Draw the cube edges:
            # Draw bottom face edges.
            for i in range(4):
                pt1 = tuple(cube_img[i])
                pt2 = tuple(cube_img[(i+1) % 4])
                cv.line(img, pt1, pt2, (255, 0, 255), 2)
            # Draw top face edges.
            for i in range(4, 8):
                pt1 = tuple(cube_img[i])
                pt2 = tuple(cube_img[4 + (i+1-4) % 4])
                cv.line(img, pt1, pt2, (255, 0, 255), 2)
            # Draw vertical edges.
            for i in range(4):
                pt_bottom = tuple(cube_img[i])
                pt_top = tuple(cube_img[i+4])
                cv.line(img, pt_bottom, pt_top, (255, 0, 255), 2)

            # -----------------------------
            # COMPUTE THE TOP FACE COLOR
            # -----------------------------
            
            bottom_face_world = board_center + bottom_face
            top_face_world = board_center + top_face

            # (1) Distance → Value (V in HSV)
            # Compute the center of the cube’s top face in world coordinates.
            top_face_center = np.mean(top_face_world, axis=0).reshape(1, 3)
            # Transform the top face center to camera coordinates.
            R_mat, _ = cv.Rodrigues(rvec)
            top_face_cam = R_mat.dot(top_face_center.T) + tvec  # shape: (3, 1)
            distance = np.linalg.norm(top_face_cam)  # distance in the same unit as square_size
            # Map distance: 0 → 255; >=400 (e.g., 400 cm) → 0, scaling linearly.
            max_distance = 400.0
            v_value = 255 * (1 - min(distance, max_distance) / max_distance)
            v_value = int(np.clip(v_value, 0, 255))
            
            # (2) Tilt → Saturation (S in HSV)
            # Compute the chessboard’s normal in world coordinates ([0, 0, 1]) transformed into camera coordinates.
            board_normal_world = np.array([0, 0, 1], dtype=np.float32).reshape(3, 1)
            board_normal_cam = R_mat.dot(board_normal_world).flatten()
            # The camera’s viewing direction is along the Z axis [0, 0, 1].
            # Compute the angle (in degrees) between board_normal_cam and [0, 0, 1].
            cos_angle = board_normal_cam[2] / np.linalg.norm(board_normal_cam)
            tilt_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            # When tilt is 0° (i.e. board is parallel to the camera), saturation = 255.
            # When tilt is 45° or more, saturation = 0.
            max_tilt = 45.0
            s_value = 255 * (1 - min(tilt_angle, max_tilt) / max_tilt)
            s_value = int(np.clip(s_value, 0, 255))
            
            # (3) Horizontal position → Hue (H in HSV)
            # Project the board center (world coordinate) to image coordinates.
            board_center_img, _ = cv.projectPoints(board_center, rvec, tvec, cameraMatrix, dist)
            board_center_img = board_center_img.reshape(-1, 2)
            x_center = board_center_img[0, 0]
            hue = int((x_center / img_width) * 180)  # OpenCV uses H in [0, 180]
            
            # Combine HSV components and convert to BGR for drawing.
            hsv_color = np.uint8([[[hue, s_value, v_value]]])
            bgr_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)[0][0]
            bgr_color = tuple(int(c) for c in bgr_color)
            
            # -----------------------------
            # FILL THE TOP FACE OF THE CUBE
            # -----------------------------
            # Use the projected points for the top face (vertices 4-7).
            top_face_pts = cube_img[4:8].reshape((-1, 1, 2))
            cv.drawChessboardCorners(img, (9,6), corners2, ret)
            cv.fillConvexPoly(img, top_face_pts, bgr_color)
            
            cv.imshow("Computer Vision - Assignment 1", img)
            cv.waitKey(500)
            cv.destroyAllWindows()

cv.destroyAllWindows()