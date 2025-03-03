import numpy as np
import cv2 as cv
import glob


# Chessboard parameters
chessboard_size = (6, 8)  # Internal corners
square_size = 115  # Square size in mm

# Criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# Prepare 3D object points
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Storage for object and image points
objpoints = []  # 3D points
imgpoints = []  # 2D points

cam_file = "cam4"

def Main():
    images = GetImages("Assignment 2/data/" + cam_file + "/intrinsics_screenshots/*.png")
    if not images:
        print("No images found! Check the folder path.")
        return
    print("Calibration in progress.")
    InitialCalibration(images, True)
    

def GetImages(FilePath: str):
    return glob.glob(FilePath)

def select_corners(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(param['corners']) < 4:
            param['corners'].append((x, y))
            print(f"Corner {len(param['corners'])} selected at ({x}, {y})")
            if len(param['corners']) == 4:
                print("All 4 corners selected.")

def InitialCalibration(images: list[str], showResults: bool):
    img = cv.imread(images[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # (width, height)

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

            manual_corners = []
            
            cv.namedWindow('img')
            cv.setMouseCallback('img', select_corners, {'corners': manual_corners})

            while len(manual_corners) < 4:
                cv.imshow('img', img)
                cv.waitKey(1)
            
            manual_corners = np.array(manual_corners, dtype=np.float32)

            dst_points = np.array([
                [0, 0],  
                [chessboard_size[1] - 1, 0],  
                [chessboard_size[1] - 1, chessboard_size[0] - 1],  
                [0, chessboard_size[0] - 1]  
            ], dtype=np.float32) * square_size
            H, _ = cv.findHomography(manual_corners, dst_points)

            x_grid, y_grid = np.meshgrid(range(chessboard_size[1]), range(chessboard_size[0]))
            grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T.astype(np.float32) * square_size

            projected_corners = cv.perspectiveTransform(grid_points.reshape(1, -1, 2), np.linalg.inv(H))
            projected_corners = projected_corners.reshape(-1, 1, 2)

            objpoints.append(objp)
            imgpoints.append(projected_corners)

            for point in projected_corners:
                cv.circle(img, tuple(point.ravel().astype(int)), 5, (0, 255, 0), -1)

            cv.imshow("img", img)
            cv.waitKey(500)

    cv.destroyAllWindows()

    img = cv.imread(images[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # (width, height)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    print(f" Final Reprojection Error: {ret}")
    print(f" Camera Matrix:\n{mtx}")
    print(f" Distortion Coefficients:\n{dist}")

    # Save to XML
    file_path = "Assignment 2/data/" + cam_file + "/intrinsics.xml"
    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_WRITE)

    fs.write("CameraMatrix", mtx)

    fs.write("DistortionCoeffs", dist)

    rvec_fixed = np.array(rvecs[0]).reshape(3, 1)
    tvec_fixed = np.array(tvecs[0]).reshape(3, 1)
    fs.write("rvec", rvec_fixed)
    fs.write("tvec", tvec_fixed)

    fs.release()
    print(f"Camera intrinsics saved to {file_path}")

if __name__ == "__main__":
    Main()