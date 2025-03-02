import numpy as np
import cv2 as cv
import glob

# Chessboard parameters
chessboard_size = (9, 6)  # Internal corners
square_size = 2.5  # Square size in cm

# Criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# Prepare 3D object points
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Storage for object and image points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Main function controls the flow
def Main():
    images = GetImages("images for run 1/*.png")
    if not images:
        print("No images found! Check the folder path.")
        return
    print("Calibration in progress.")
    InitialCalibration(images, True)
    print("Calibration Done! Creating cube on the chessboard.")
    CreateCubeOnTheChessboard(GetImages("testImage.png"))

def GetImages(FilePath: str):
    return glob.glob(FilePath)

# Calibration
def InitialCalibration(images: list[str], showResults: bool):
    img = cv.imread(images[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    #Calibration
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if showResults:
                cv.drawChessboardCorners(img, chessboard_size, corners, ret)
                cv.imshow("img", img)
                cv.waitKey(200)
        else:
            print(f"Chessboard not found in {fname}, please select the 4 corners of the chessboard starting from top left and ending at bottom left.")

            # Manually assign the corners
            manual_corners = []
            
            cv.namedWindow('img')
            cv.setMouseCallback('img', select_corners, {'corners': manual_corners})

            while len(manual_corners) < 4:
                cv.imshow('img', img)
                cv.waitKey(1)
            
            # Convert the selected corners to NumPy array
            manual_corners = np.array(manual_corners, dtype=np.float32)

            # Define object points (real world coordinates in cm)
            dst_points = np.array([
                [0, 0],  
                [chessboard_size[1] - 1, 0],  
                [chessboard_size[1] - 1, chessboard_size[0] - 1],  
                [0, chessboard_size[0] - 1]  
            ], dtype=np.float32) * square_size
            H, _ = cv.findHomography(manual_corners, dst_points)

            # Generate grid of expected inner corners
            x_grid, y_grid = np.meshgrid(range(chessboard_size[1]), range(chessboard_size[0]))
            grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T.astype(np.float32) * square_size

            # Transform these points to image space
            projected_corners = cv.perspectiveTransform(grid_points.reshape(1, -1, 2), np.linalg.inv(H))
            projected_corners = projected_corners.reshape(-1, 1, 2)

            # Store points
            objpoints.append(objp)
            imgpoints.append(projected_corners)

            # Draw selected and calculated points
            for point in projected_corners:
                cv.circle(img, tuple(point.ravel().astype(int)), 5, (0, 255, 0), -1)

            cv.imshow("img", img)
            cv.waitKey(500)

    cv.destroyAllWindows()

# Function to manually select the corners
def select_corners(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(param['corners']) < 4:
            param['corners'].append((x, y))
            print(f"Corner {len(param['corners'])} selected at ({x}, {y})")
            if len(param['corners']) == 4:
                print("All 4 corners selected.")

# Create a Cube on the chessboard
def CreateCubeOnTheChessboard(images: list[str]):
    # Get image size, Camera Matrix & Distortion
    img = cv.imread(images[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    cameraMatrix = mtx
    camereaDistortion = dist
    print(f" Final Reprojection Error: {ret}")
    print(f" Camera Matrix:\n{cameraMatrix}")
    print(f" Distortion Coefficients:\n{camereaDistortion}")
    print(f" size:\n{image_size}")

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if ret == True:
            cornersForSolvePnP = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

            isSolvePnPWorks, rvec, tvec = cv.solvePnP(objp, cornersForSolvePnP, cameraMatrix, camereaDistortion)
            if isSolvePnPWorks == True:
                board_center = np.mean(objp, axis=0).reshape(1, 3)
                axis_length = square_size * 2  #length equal to 2 squares
                axes_points = np.float32([
                    [0, 0, 0],
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, -axis_length]
                ])
                # Place the axes at the board center.
                axes_world = board_center + axes_points
                axes_img, _ = cv.projectPoints(axes_world, rvec, tvec, cameraMatrix, dist)
                axes_img = axes_img.reshape(-1, 2).astype(int)

                # Draw the axes lines on the image.
                origin_pt = tuple(axes_img[0])
                cv.line(img, origin_pt, tuple(axes_img[1]), (0, 0, 255), 3)# X-axis
                cv.line(img, origin_pt, tuple(axes_img[2]), (0, 255, 0), 3)#Y-axis
                cv.line(img, origin_pt, tuple(axes_img[3]), (255, 0, 0), 3)#Z-axis

                # Box Position = bottom-left (3,2), bottom-right (5,2), top-right (5,4), and top-left (3,4)
                bottom_face = np.float32([
                    [3 * square_size, 2 * square_size, 0],
                    [5 * square_size, 2 * square_size, 0],
                    [5 * square_size, 4 * square_size, 0],
                    [3 * square_size, 4 * square_size, 0]
                ])

                cube_height = 2 * square_size # To create 2x2x2 cube
                top_face = bottom_face.copy()
                top_face[:, 2] = -cube_height

                cube_world_points = np.concatenate((bottom_face, top_face), axis=0)
                cube_img, _ = cv.projectPoints(cube_world_points, rvec, tvec, cameraMatrix, camereaDistortion)
                cube_img = cube_img.reshape(-1, 2).astype(int)

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
                
                top_face_pts = cube_img[4:8].reshape((-1, 1, 2))


                # Colouring of the top polygon
                bottom_face_world = board_center + bottom_face
                top_face_world = board_center + top_face
                # Compute the center of the cube’s top face in world coordinates.
                top_face_center = np.mean(top_face_world, axis=0).reshape(1, 3)
                # Transform the top face center to camera coordinates.
                R_mat, _ = cv.Rodrigues(rvec)
                top_face_cam = R_mat.dot(top_face_center.T) + tvec
                distance = np.linalg.norm(top_face_cam)

                max_distance = 400.0
                v_value = 255 * (1 - min(distance, max_distance) / max_distance)
                v_value = int(np.clip(v_value, 0, 255))
                
                # Compute the chessboard’s normal in world coordinates ([0, 0, 1]) transformed into camera coordinates.
                board_normal_world = np.array([0, 0, 1], dtype=np.float32).reshape(3, 1)
                board_normal_cam = R_mat.dot(board_normal_world).flatten()
                # The camera’s viewing direction is along the Z axis [0, 0, 1].
                # Compute the angle (in degrees) between board_normal_cam and [0, 0, 1].
                cos_angle = board_normal_cam[2] / np.linalg.norm(board_normal_cam)
                tilt_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

                max_tilt = 45.0
                s_value = 255 * (1 - min(tilt_angle, max_tilt) / max_tilt)
                s_value = int(np.clip(s_value, 0, 255))
                
                # Project the board center (world coordinate) to image coordinates.
                board_center_img, _ = cv.projectPoints(board_center, rvec, tvec, cameraMatrix, dist)
                board_center_img = board_center_img.reshape(-1, 2)
                x_center = board_center_img[0, 0]
                hue = int((x_center / image_size[1]) * 180)  # OpenCV uses H in [0, 180]
                
                # Combine HSV components and convert to BGR for drawing.
                hsv_color = np.uint8([[[hue, s_value, v_value]]])
                bgr_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)[0][0]
                bgr_color = tuple(int(c) for c in bgr_color)

                # Use the projected points for the top face (vertices 4-7).
                top_face_pts = cube_img[4:8].reshape((-1, 1, 2))
                cv.fillConvexPoly(img, top_face_pts, bgr_color)
                
                cv.imshow("Computer Vision - Assignment 1", img)
                cv.waitKey(0)
                cv.destroyAllWindows()

    cv.destroyAllWindows()

if __name__ == "__main__":
    Main()