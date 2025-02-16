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
    images = GetImageList("images for run 2/*.png")
    if not images:
        print("No images found! Check the folder path.")
        return
    print("Calibration in progress.")
    InitialCalibration(images)
    print("Calibration Done! Creating cube on the chessboard.")
    CreateCubeOnTheChessboard(images)

def GetImageList(FilePath: str):
    return glob.glob(FilePath)

def InitialCalibration(images: list[str]):
    # **Step 1: First Pass - Initial Calibration**
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            cv.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv.imshow("Initial Detection", img)
            cv.waitKey(200)

    cv.destroyAllWindows()

def CreateCubeOnTheChessboard(images: list[str]):
    # Get image size & Get Camera Matrix and Distance
    img = cv.imread(images[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # (width, height)
    _, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    cameraMatrix = mtx
    camereaDistance = dist

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if ret == True:
            cornersForSolvePnP = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

            isSolvePnPWorks, rvec, tvec = cv.solvePnP(objp, cornersForSolvePnP, cameraMatrix, camereaDistance)
            if isSolvePnPWorks == True:
                # bottom-left (3,2), bottom-right (5,2), top-right (5,4), and top-left (3,4)
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
                cube_img, _ = cv.projectPoints(cube_world_points, rvec, tvec, cameraMatrix, camereaDistance)
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

                # Draw the cube and corners on the chessboard
                cv.drawChessboardCorners(img, chessboard_size, corners, ret)
                cv.fillConvexPoly(img, top_face_pts, red_color)
                
                cv.imshow("Computer Vision - Assignment 1", img)
                cv.waitKey(500)
                cv.destroyAllWindows()

    cv.destroyAllWindows()

if __name__ == "__main__":
    Main()