import numpy as np
import cv2 as cv
import glob

# Chessboard parameters
chessboard_size = (6, 8)  # number of inner corners per chessboard row and column
square_size = 115  # square size in mm

# Criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# Prepare 3D object points for the full chessboard (all inner corners)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Storage for object and image points
objpoints = []  # 3D world points
imgpoints = []  # 2D image points

cam_file = "cam4"

def draw_axes_on_chessboard(image, mtx, dist, rvec, tvec, square_size):
    """
    Draws the X, Y, Z axes on the image starting from the origin (top-left corner of the chessboard).
    X axis (red), Y axis (green), Z axis (blue). The axes lengths are set to 3*square_size.
    """
    # Define 3D points for the axes endpoints
    origin = np.array([[0, 0, 0]], dtype=np.float32)
    x_axis = np.array([[3 * square_size, 0, 0]], dtype=np.float32)
    y_axis = np.array([[0, 3 * square_size, 0]], dtype=np.float32)
    # For visualization, we often choose a negative z-axis so that it appears to come "out of" the board.
    z_axis = np.array([[0, 0, -3 * square_size]], dtype=np.float32)

    # Project these 3D points to the image plane
    imgpts_origin, _ = cv.projectPoints(origin, rvec, tvec, mtx, dist)
    imgpts_x, _ = cv.projectPoints(x_axis, rvec, tvec, mtx, dist)
    imgpts_y, _ = cv.projectPoints(y_axis, rvec, tvec, mtx, dist)
    imgpts_z, _ = cv.projectPoints(z_axis, rvec, tvec, mtx, dist)

    origin_pt = tuple(imgpts_origin.ravel().astype(int))
    x_pt = tuple(imgpts_x.ravel().astype(int))
    y_pt = tuple(imgpts_y.ravel().astype(int))
    z_pt = tuple(imgpts_z.ravel().astype(int))

    # Draw the axes lines
    cv.line(image, origin_pt, x_pt, (0, 0, 255), 3)  # X-axis in red
    cv.line(image, origin_pt, y_pt, (0, 255, 0), 3)  # Y-axis in green
    cv.line(image, origin_pt, z_pt, (255, 0, 0), 3)  # Z-axis in blue

    # Label the axes
    cv.putText(image, "X", x_pt, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv.putText(image, "Y", y_pt, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.putText(image, "Z", z_pt, cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return image

def Main():
    images = GetImages("Assignment 2/data/" + cam_file + "/intrinsics_screenshots/*.png")
    if not images:
        print("No images found! Check the folder path.")
        return

    print("Calibration in progress.")
    calib_result = InitialCalibration(images, True)
    if calib_result is None:
        print("Calibration failed.")
        return
    mtx, dist, rvecs, tvecs, image_size = calib_result

    # Load one calibration image to visualize the axes on the chessboard.
    img = cv.imread(images[0])
    if img is None:
        print("Could not load image for drawing axes.")
        return

    # Use the extrinsic parameters from the first calibration image.
    rvec = rvecs[0]
    tvec = tvecs[0]

    # Draw the 3D axes on the image.
    img_with_axes = draw_axes_on_chessboard(img.copy(), mtx, dist, rvec, tvec, square_size)
    cv.imshow("Calibration with Axes", img_with_axes)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save the result image to file.
    result_path = "Assignment 2/data/" + cam_file + "/calibration_with_axes.png"
    cv.imwrite(result_path, img_with_axes)
    print(f"Calibration image with axes saved to {result_path}")

def GetImages(filePath: str):
    return glob.glob(filePath)

def select_corners(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(param['corners']) < 4:
            param['corners'].append((x, y))
            print(f"Corner {len(param['corners'])} selected at ({x}, {y})")
            if len(param['corners']) == 4:
                print("All 4 corners selected.")

def InitialCalibration(images: list, showResults: bool):
    # Load the first image to determine image size.
    img = cv.imread(images[0])
    if img is None:
        print("Failed to load the first image.")
        return None
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # (width, height)

    # Perform calibration for each image.
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            continue
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
            print(f"Chessboard not found in {fname}. Please manually select 4 corners starting from the top-left.")
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

    # Calibrate the camera.
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    print(f"Final Reprojection Error: {ret}")
    print(f"Camera Matrix:\n{mtx}")
    print(f"Distortion Coefficients:\n{dist}")

    # Save the intrinsics to an XML file.
    file_path = "Assignment 2/data/" + cam_file + "/intrinsics.xml"
    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_WRITE)
    fs.write("CameraMatrix", mtx)
    fs.write("DistortionCoeffs", dist)
    fs.release()
    print(f"Camera intrinsics saved to {file_path}")

    return mtx, dist, rvecs, tvecs, image_size

if __name__ == "__main__":
    Main()
