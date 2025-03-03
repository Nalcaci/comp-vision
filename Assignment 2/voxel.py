import numpy as np
import cv2 as cv
import glob
import os


# -----------------------------
# Task 2: Background Subtraction
# -----------------------------

def build_background_model(video_path, num_frames=30):
    """
    Reads a number of frames from a background video, converts them to HSV, and computes their average.
    Returns the background model (as an HSV image).
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video file {video_path} cannot be opened.")
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
    _, mask_h = cv.threshold(diff[:,:,0], thresholds[0], 255, cv.THRESH_BINARY)
    _, mask_s = cv.threshold(diff[:,:,1], thresholds[1], 255, cv.THRESH_BINARY)
    _, mask_v = cv.threshold(diff[:,:,2], thresholds[2], 255, cv.THRESH_BINARY)
    mask = cv.bitwise_and(mask_h, mask_s)
    mask = cv.bitwise_and(mask, mask_v)
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
    Returns a list of voxels that are "on" (each voxel is [x, y, z]).
    """
    voxels_on = []
    x_range = np.arange(0, 128, voxel_step)
    y_range = np.arange(0, 64, voxel_step)
    z_range = np.arange(0, 128, voxel_step)
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
    
    # === Task 2: Background Subtraction ===
    cam_ids = ['cam1', 'cam2', 'cam3', 'cam4']
    base_path = os.path.join("Assignment 2", "data")
    calibration_params = {}  # To store (mtx, dist, rvec, tvec) per camera
    foreground_masks = {}    # To store one foreground mask per camera (for a chosen frame)
    
    # For each camera, load calibration parameters (from config.xml if available, else intrinsics.xml)
    for cam in cam_ids:
        cam_path = os.path.join(base_path, cam)
        config_file = os.path.join(cam_path, "intrinsics.xml")
        fs = cv.FileStorage(config_file, cv.FILE_STORAGE_READ)
        if not fs.isOpened():
            print(f"Failed to open {config_file} for {cam}.")
            continue
        mtx = fs.getNode("CameraMatrix").mat()
        dist = fs.getNode("DistortionCoeffs").mat()
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
        if not cap.isOpened():
            print(f"Failed to open video for {cam}.")
            continue
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Failed to read video frame for {cam}.")
            continue
        mask = get_foreground_mask(frame, bg_model, thresholds=(15, 30, 30))
        foreground_masks[cam] = mask
        
        cv.imshow(f"Foreground Mask - {cam}", mask)
        cv.waitKey(200)
    cv.destroyAllWindows()
    
    if len(calibration_params) < 4:
        print("Not all cameras have valid calibration parameters. Aborting voxel reconstruction.")
        return

    # === Task 3: Voxel Reconstruction ===
    voxels_on = voxel_reconstruction(foreground_masks, calibration_params, voxel_step=8)
    print(f"Number of voxels reconstructed: {len(voxels_on)}")
    
    # Integration with the visualization module (if available):
    # For example:
    # from visualization_module import display_voxels
    # display_voxels(voxels_on)
    
if __name__ == "__main__":
    main()
