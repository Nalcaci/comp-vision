import numpy as np
import cv2 as cv
import glob
import os

# -----------------------------
# Task 1: Calibration and 3D axes
# -----------------------------

# Chessboard parameters
chessboard_size = (6, 8)  # Internal corners (rows, columns)
square_size = 115         # Square size in mm

# Criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# Prepare 3D object points for the chessboard (all inner corners)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Storage for object and image points (used during calibration)
objpoints = []  # 3D world points
imgpoints = []  # 2D image points

def draw_axes_on_chessboard(image, mtx, dist, rvec, tvec, square_size):
    """
    Projects 3D axes endpoints (X, Y, Z) starting at the checkerboard's origin
    and draws them on the image.
    X axis: red, Y axis: green, Z axis: blue.
    """
    origin = np.array([[0, 0, 0]], dtype=np.float32)
    x_axis = np.array([[3 * square_size, 0, 0]], dtype=np.float32)
    y_axis = np.array([[0, 3 * square_size, 0]], dtype=np.float32)
    z_axis = np.array([[0, 0, -3 * square_size]], dtype=np.float32)  # negative so it appears coming "out" of the board

    imgpts_origin, _ = cv.projectPoints(origin, rvec, tvec, mtx, dist)
    imgpts_x, _ = cv.projectPoints(x_axis, rvec, tvec, mtx, dist)
    imgpts_y, _ = cv.projectPoints(y_axis, rvec, tvec, mtx, dist)
    imgpts_z, _ = cv.projectPoints(z_axis, rvec, tvec, mtx, dist)

    origin_pt = tuple(imgpts_origin.ravel().astype(int))
    x_pt = tuple(imgpts_x.ravel().astype(int))
    y_pt = tuple(imgpts_y.ravel().astype(int))
    z_pt = tuple(imgpts_z.ravel().astype(int))

    cv.line(image, origin_pt, x_pt, (0, 0, 255), 3)  # X-axis in red
    cv.line(image, origin_pt, y_pt, (0, 255, 0), 3)  # Y-axis in green
    cv.line(image, origin_pt, z_pt, (255, 0, 0), 3)  # Z-axis in blue

    cv.putText(image, "X", x_pt, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv.putText(image, "Y", y_pt, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.putText(image, "Z", z_pt, cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return image

def select_corners(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(param['corners']) < 4:
            param['corners'].append((x, y))
            print(f"Corner {len(param['corners'])} selected at ({x}, {y})")
            if len(param['corners']) == 4:
                print("All 4 corners selected.")

def calibrate_camera_from_images(images, showResults=True):
    """
    Given a list of calibration images, detects the chessboard corners (or lets the user select them)
    and computes the camera intrinsics and extrinsics.
    Saves the intrinsics (and optionally extrinsics) into an XML file.
    Returns (mtx, dist, rvecs, tvecs, image_size).
    """
    global objpoints, imgpoints
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            if showResults:
                cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
                cv.imshow("Calibration", img)
                cv.waitKey(200)
        else:
            print(f"Chessboard not automatically detected in {fname}.")
            manual_corners = []
            cv.namedWindow('Calibration')
            cv.setMouseCallback('Calibration', select_corners, {'corners': manual_corners})
            while len(manual_corners) < 4:
                cv.imshow('Calibration', img)
                cv.waitKey(1)
            manual_corners = np.array(manual_corners, dtype=np.float32)
            dst_points = np.array([
                [0, 0],
                [chessboard_size[1]-1, 0],
                [chessboard_size[1]-1, chessboard_size[0]-1],
                [0, chessboard_size[0]-1]
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
            cv.imshow("Calibration", img)
            cv.waitKey(500)
    cv.destroyAllWindows()
    # Use the first image to get the image size.
    sample_img = cv.imread(images[0])
    gray = cv.cvtColor(sample_img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    print(f"Reprojection Error: {ret}")
    print(f"Camera Matrix:\n{mtx}")
    print(f"Distortion Coefficients:\n{dist}")

    # Save calibration to XML (for example, in cam4 directory)
    file_path = os.path.join("Assignment 2", "data", "cam4", "intrinsics.xml")
    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_WRITE)
    fs.write("CameraMatrix", mtx)
    fs.write("DistortionCoeffs", dist)
    # Optionally save the first extrinsics (rvec and tvec) for visualization:
    fs.write("rvec", rvecs[0])
    fs.write("tvec", tvecs[0])
    fs.release()
    print(f"Calibration parameters saved to {file_path}")

    return mtx, dist, rvecs, tvecs, image_size

# -----------------------------
# Task 2: Background Subtraction
# -----------------------------

def build_background_model(video_path, num_frames=30):
    """
    Reads a number of frames from a background video, converts them to HSV, and computes their average.
    Returns the background model (as an HSV image).
    """
    cap = cv.VideoCapture(video_path)
    frames = []
    count = 0
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frames.append(frame_hsv.astype(np.float32))
        count += 1
    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames read from background video.")
    background_model = np.mean(frames, axis=0).astype(np.uint8)
    return background_model

def get_foreground_mask(frame, background_model, thresholds=(15, 30, 30)):
    """
    Converts a frame to HSV and computes its absolute difference to the background model.
    Applies per-channel thresholds (for H, S, V) and combines the results to produce a binary mask.
    Morphological operations help reduce noise.
    """
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    diff = cv.absdiff(frame_hsv, background_model)
    # Threshold differences for each channel:
    _, mask_h = cv.threshold(diff[:,:,0], thresholds[0], 255, cv.THRESH_BINARY)
    _, mask_s = cv.threshold(diff[:,:,1], thresholds[1], 255, cv.THRESH_BINARY)
    _, mask_v = cv.threshold(diff[:,:,2], thresholds[2], 255, cv.THRESH_BINARY)
    # Combine channels: a pixel is foreground if all three thresholds are exceeded.
    mask = cv.bitwise_and(mask_h, mask_s)
    mask = cv.bitwise_and(mask, mask_v)
    # Optional post-processing to remove noise:
    kernel = np.ones((3,3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return mask

# -----------------------------
# Task 3: Voxel Reconstruction
# -----------------------------

def voxel_reconstruction(foreground_masks, calibration_params, voxel_step=8):
    """
    Given a dictionary of foreground masks (one per camera) and the corresponding calibration parameters,
    this function iterates over a predefined voxel grid (a half-cube of size 128x64x128, with 'y' as up)
    and uses projectPoints() to map each voxel into the image planes.
    A voxel is considered part of the reconstructed volume if its projection lands on a foreground pixel
    in every camera.
    The voxel grid is sampled with the provided voxel_step to speed-up computation.
    Returns a list of voxels that are "on" (each voxel is [x, y, z]).
    """
    voxels_on = []
    # Define voxel grid boundaries.
    x_range = np.arange(0, 128, voxel_step)
    y_range = np.arange(0, 64, voxel_step)
    z_range = np.arange(0, 128, voxel_step)
    # Note: In a production setting, consider vectorizing this loop.
    for x in x_range:
        for y in y_range:
            for z in z_range:
                point_3d = np.array([[x, y, z]], dtype=np.float32)
                valid = True
                for cam_id, params in calibration_params.items():
                    mtx, dist, rvec, tvec = params
                    imgpt, _ = cv.projectPoints(point_3d, rvec, tvec, mtx, dist)
                    imgpt = imgpt.ravel().astype(int)
                    mask = foreground_masks[cam_id]
                    h, w = mask.shape
                    if imgpt[0] < 0 or imgpt[0] >= w or imgpt[1] < 0 or imgpt[1] >= h:
                        valid = False
                        break
                    if mask[imgpt[1], imgpt[0]] == 0:
                        valid = False
                        break
                if valid:
                    voxels_on.append([x, y, z])
    return voxels_on

# -----------------------------
# Integration and Main Routine
# -----------------------------

def main():
    # === Task 1: Calibration and 3D Axes Visualization ===
    # For example, calibrate camera from screenshots (e.g., for cam4).
    cam_dir = os.path.join("Assignment 2", "data", "cam4")
    image_path = os.path.join(cam_dir, "intrinsics_screenshots", "*.png")
    images = glob.glob(image_path)
    if len(images) == 0:
        print("No calibration images found. Check the path.")
    else:
        mtx, dist, rvecs, tvecs, image_size = calibrate_camera_from_images(images, showResults=True)
        # Load one calibration image to overlay the 3D axes.
        img = cv.imread(images[0])
        img_with_axes = draw_axes_on_chessboard(img.copy(), mtx, dist, rvecs[0], tvecs[0], square_size)
        cv.imshow("Calibration with 3D Axes", img_with_axes)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    # === Task 2: Background Subtraction ===
    # Process each camera (assume 4 cameras) for background modeling and foreground extraction.
    cam_ids = ['cam1', 'cam2', 'cam3', 'cam4']
    base_path = os.path.join("Assignment 2", "data")
    calibration_params = {}     # To store (mtx, dist, rvec, tvec) per camera
    foreground_masks = {}         # To store one foreground mask per camera (for a chosen frame)
    
    # For each camera, load calibration parameters (from config.xml if available, else intrinsics.xml)
    for cam in cam_ids:
        cam_path = os.path.join(base_path, cam)
        # Prefer config.xml (which should contain extrinsics) over intrinsics.xml.
        config_file = os.path.join(cam_path, "config.xml")
        if not os.path.exists(config_file):
            config_file = os.path.join(cam_path, "intrinsics.xml")
        fs = cv.FileStorage(config_file, cv.FILE_STORAGE_READ)
        mtx = fs.getNode("CameraMatrix").mat()
        dist = fs.getNode("DistortionCoeffs").mat()
        # Extrinsics (rvec and tvec) are expected to be saved after calibration.
        rvec = fs.getNode("rvec").mat() if not fs.getNode("rvec").empty() else None
        tvec = fs.getNode("tvec").mat() if not fs.getNode("tvec").empty() else None
        fs.release()
        if rvec is None or tvec is None:
            print(f"Extrinsics not found for {cam}. Run calibration for {cam} first.")
            continue
        calibration_params[cam] = (mtx, dist, rvec, tvec)
        
        # Build background model using background.avi
        background_video = os.path.join(cam_path, "background.avi")
        try:
            bg_model = build_background_model(background_video, num_frames=30)
        except Exception as e:
            print(f"Error building background model for {cam}: {e}")
            continue
        
        # For demonstration, process the first frame of video.avi to extract the foreground mask.
        video_path = os.path.join(cam_path, "video.avi")
        cap = cv.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Failed to read video frame for {cam}.")
            continue
        mask = get_foreground_mask(frame, bg_model, thresholds=(15, 30, 30))
        foreground_masks[cam] = mask
        
        # Optionally, display the foreground mask for each camera:
        cv.imshow(f"Foreground Mask - {cam}", mask)
        cv.waitKey(200)
    cv.destroyAllWindows()
    
    if len(calibration_params) < 4:
        print("Not all cameras have valid calibration parameters. Aborting voxel reconstruction.")
        return

    # === Task 3: Voxel Reconstruction ===
    # For simplicity, here we reconstruct voxels for the current frame (one foreground mask per camera).
    # The voxel grid is defined as a half-cube (side lengths 128, height 64) with a sampling step.
    voxels_on = voxel_reconstruction(foreground_masks, calibration_params, voxel_step=8)
    print(f"Number of voxels reconstructed: {len(voxels_on)}")
    
    # Integration with the visualization module:
    # At this point you would hand over the voxel list (voxels_on) to the 3D visualization routines
    # provided in the repository (e.g., through a function like display_voxels(voxels_on)).
    # For example:
    # from visualization_module import display_voxels
    # display_voxels(voxels_on)
    # (The actual function names and integration depend on the repository's code structure.)
    
if __name__ == "__main__":
    main()