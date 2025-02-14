import numpy as np
import cv2 as cv
import glob

pattern_size = (9, 6)  # (columns, rows)
square_size = 2.5

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*9,3), np.float32)
# objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2) * 2.5 #the squares are 2.5 cm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

distCoeffs = np.zeros((5, 1))  # assuming no lens distortion

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
        # objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        # imgpoints.append(corners2)

        # # Draw and display the corners
        # cv.drawChessboardCorners(img, (9,6), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)

        img_height, img_width = img.shape[:2]
        cameraMatrix = np.array([[1000,    0, img_width/2],
                                [   0, 1000, img_height/2],
                                [   0,    0,         1]], dtype=np.float64)
        
        objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
        # The grid: x varies along the number of columns, y varies along the number of rows.
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        ret2, rvec, tvec = cv.solvePnP(objp, corners2, cameraMatrix, distCoeffs)
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
            axes_img, _ = cv.projectPoints(axes_world, rvec, tvec, cameraMatrix, distCoeffs)
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
            # bottom_face = np.float32([
            #     [-half, -half, 0],
            #     [ half, -half, 0],
            #     [ half,  half, 0],
            #     [-half,  half, 0]
            # ])

            # Define the indices for the four outer corners:
            top_left     = objp[4]
            top_right    = objp[2]         # (9-1) because indexing starts at 0
            bottom_right = objp[-2]
            bottom_left  = objp[-4]        # Last point of the second-to-last row

            # Create the bottom face using these corners.
            bottom_face = np.float32([top_left, top_right, bottom_right, bottom_left])

            # Define top face vertices (shifted in z by -cube_size)
            # top_face = bottom_face.copy()
            # top_face[:, 2] = -cube_size

            cube_size = square_size * 2  # or any desired cube size
            top_face = bottom_face.copy()
            top_face[:, 2] = -cube_size  # shift upward (negative z direction)
            
            # Translate cube vertices so that the bottom face is centered at board_center.
            bottom_face_world = board_center + bottom_face
            top_face_world = board_center + top_face
            # Combine all eight vertices (order: bottom face then top face)
            # cube_world = np.concatenate((bottom_face_world, top_face_world), axis=0)
            # cube_img, _ = cv.projectPoints(cube_world, rvec, tvec, cameraMatrix, distCoeffs)
            # cube_img = cube_img.reshape(-1, 2).astype(int)
            cube_world = np.concatenate((bottom_face, top_face), axis=0)
            cube_img, _ = cv.projectPoints(cube_world, rvec, tvec, cameraMatrix, distCoeffs)
            cube_img = cube_img.reshape(-1, 2).astype(int)
            
            # Draw the cube edges:
            # Bottom face edges
            for i in range(4):
                pt1 = tuple(cube_img[i])
                pt2 = tuple(cube_img[(i+1) % 4])
                cv.line(img, pt1, pt2, (255, 0, 255), 2)
            # Top face edges
            for i in range(4, 8):
                pt1 = tuple(cube_img[i])
                pt2 = tuple(cube_img[4 + (i+1-4) % 4])
                cv.line(img, pt1, pt2, (255, 0, 255), 2)
            # Vertical edges
            for i in range(4):
                pt_bottom = tuple(cube_img[i])
                pt_top = tuple(cube_img[i+4])
                cv.line(img, pt_bottom, pt_top, (255, 0, 255), 2)

            # -----------------------------
            # COMPUTE THE TOP FACE COLOR
            # -----------------------------
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
            board_center_img, _ = cv.projectPoints(board_center, rvec, tvec, cameraMatrix, distCoeffs)
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
            
            # -----------------------------
            # DISPLAY THE AUGMENTED IMAGE
            # -----------------------------
            cv.imshow("Augmented Reality", img)
            cv.waitKey(0)
            cv.destroyAllWindows()
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