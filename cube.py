import numpy as np
import cv2 as cv

def main():
    # -----------------------------
    # PARAMETERS & CALIBRATION DATA
    # -----------------------------
    # Chessboard pattern (number of inner corners per row and column)
    # (Here, 9 corners horizontally and 6 vertically.)
    pattern_size = (9, 6)  # (columns, rows)
    square_size = 2.5      # in your chosen unit (e.g., centimeters)
    
    # Termination criteria for cornerSubPix refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Dummy calibration parameters.
    # Replace these with your real camera calibration results.
    # For this example, we assume a focal length of 1000 and the principal point at the image center.
    # (Note: tvec units will match the units of square_size.)
    test_img = cv.imread("images/c14.png")
    if test_img is None:
        print("Error: Could not load test image.")
        return
    img_height, img_width = test_img.shape[:2]
    
    cameraMatrix = np.array([[1000,    0, img_width/2],
                             [   0, 1000, img_height/2],
                             [   0,    0,         1]], dtype=np.float64)
    distCoeffs = np.zeros((5, 1))  # assuming no lens distortion
    
    # -----------------------------
    # FIND CHESSBOARD CORNERS
    # -----------------------------
    gray = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        print("Error: Chessboard corners not found in test image!")
        return
    # Refine corner locations
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # -----------------------------
    # PREPARE OBJECT POINTS
    # -----------------------------
    # Create the world coordinates for the chessboard corners.
    # We assume the chessboard is on the plane z = 0 with the top-left corner at (0,0,0).
    # Note: Use (columns, rows) order.
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    # The grid: x varies along the number of columns, y varies along the number of rows.
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # -----------------------------
    # ESTIMATE POSE USING solvePnP
    # -----------------------------
    ret, rvec, tvec = cv.solvePnP(objp, corners, cameraMatrix, distCoeffs)
    if not ret:
        print("Error: solvePnP failed!")
        return

    # Compute the chessboard (board) center in world coordinates.
    board_center = np.mean(objp, axis=0).reshape(1, 3)
    
    # -----------------------------
    # DRAW 3D AXES
    # -----------------------------
    # We want to draw axes with origin at the board center.
    # Define three axes points (offsets relative to the board center):
    axis_length = square_size * 3  # e.g., length equal to three squares
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
    cv.line(test_img, origin_pt, tuple(axes_img[1]), (0, 0, 255), 3)  # X axis in red
    cv.line(test_img, origin_pt, tuple(axes_img[2]), (0, 255, 0), 3)  # Y axis in green
    cv.line(test_img, origin_pt, tuple(axes_img[3]), (255, 0, 0), 3)  # Z axis in blue
    
    # -----------------------------
    # DEFINE THE CUBE
    # -----------------------------
    # We define a cube whose bottom face is centered at the board center (on the chessboard plane, z=0)
    # and whose top face is at z = -cube_size (extending upward).
    cube_size = axis_length  # set cube side length; adjust as desired
    half = cube_size / 2.0
    
    # Define bottom face vertices (relative to board_center)
    bottom_face = np.float32([
        [-half, -half, 0],
        [ half, -half, 0],
        [ half,  half, 0],
        [-half,  half, 0]
    ])
    # Define top face vertices (shifted in z by -cube_size)
    top_face = bottom_face.copy()
    top_face[:, 2] = -cube_size
    
    # Translate cube vertices so that the bottom face is centered at board_center.
    bottom_face_world = board_center + bottom_face
    top_face_world = board_center + top_face
    # Combine all eight vertices (order: bottom face then top face)
    cube_world = np.concatenate((bottom_face_world, top_face_world), axis=0)
    cube_img, _ = cv.projectPoints(cube_world, rvec, tvec, cameraMatrix, distCoeffs)
    cube_img = cube_img.reshape(-1, 2).astype(int)
    
    # Draw the cube edges:
    # Bottom face edges
    for i in range(4):
        pt1 = tuple(cube_img[i])
        pt2 = tuple(cube_img[(i+1) % 4])
        cv.line(test_img, pt1, pt2, (255, 0, 255), 2)
    # Top face edges
    for i in range(4, 8):
        pt1 = tuple(cube_img[i])
        pt2 = tuple(cube_img[4 + (i+1-4) % 4])
        cv.line(test_img, pt1, pt2, (255, 0, 255), 2)
    # Vertical edges
    for i in range(4):
        pt_bottom = tuple(cube_img[i])
        pt_top = tuple(cube_img[i+4])
        cv.line(test_img, pt_bottom, pt_top, (255, 0, 255), 2)
    
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
    cv.fillConvexPoly(test_img, top_face_pts, bgr_color)
    
    # -----------------------------
    # DISPLAY THE AUGMENTED IMAGE
    # -----------------------------
    cv.imshow("Augmented Reality", test_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()