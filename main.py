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

# **Step 1: First Pass - Initial Calibration**
def InitialCalibration(images: list[str], showResults: bool):
    img = cv.imread(images[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # (width, height)

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

            # Define object points (real-world coordinates in cm)
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

# **Step 2: Creating a Cube on the chessboard**
def CreateCubeOnTheChessboard(images: list[str]):
    # Get image size, Camera Matrix & Distortion
    img = cv.imread(images[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # (width, height)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    cameraMatrix = mtx
    camereaDistortion = dist
    print(f" Final Reprojection Error: {ret}")
    print(f" Camera Matrix:\n{cameraMatrix}")
    print(f" Distortion Coefficients:\n{camereaDistortion}")

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if ret == True:
            cornersForSolvePnP = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

            isSolvePnPWorks, rvec, tvec = cv.solvePnP(objp, cornersForSolvePnP, cameraMatrix, camereaDistortion)
            if isSolvePnPWorks == True:
                board_center = np.mean(objp, axis=0).reshape(1, 3)
                axis_length = square_size * 2  # e.g., length equal to three squares
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
                red_color = (0, 0, 255)

                # Draw the corners and cube on the chessboard
                cv.drawChessboardCorners(img, chessboard_size, corners, ret)
                cv.fillConvexPoly(img, top_face_pts, red_color)
                
                cv.imshow("Computer Vision - Assignment 1", img)
                cv.waitKey(0)
                cv.destroyAllWindows()

    cv.destroyAllWindows()

if __name__ == "__main__":
    Main()