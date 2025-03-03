import numpy as np
import cv2 as cv
import os

from Background_Subtraction import get_background_frame

def build_background_model(video_path, num_frames=30):

    return get_background_frame(video_path, method="median", sample_rate=10)

def get_foreground_mask(frame, bg_model, thresholds=(15, 30, 30)):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    diff = cv.absdiff(bg_model, gray_frame)
    _, mask = cv.threshold(diff, thresholds[0], 255, cv.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)
    return mask

def voxel_reconstruction(foreground_masks, calibration_params, voxel_step=8, grid_dims=(128, 64, 128), min_views=3):
    voxel_lut = {cam: {} for cam in calibration_params.keys()}
    x_range = np.arange(0, grid_dims[0], voxel_step)
    y_range = np.arange(0, grid_dims[1], voxel_step)
    z_range = np.arange(0, grid_dims[2], voxel_step)
    
    for cam_id, params in calibration_params.items():
        mtx, dist, rvec, tvec = params
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    point_3d = np.array([[x, y, z]], dtype=np.float32)
                    imgpt, _ = cv.projectPoints(point_3d, rvec, tvec, mtx, dist)
                    imgpt = imgpt.ravel().astype(int)
                    voxel_lut[cam_id][(x, y, z)] = tuple(imgpt)
    
    voxels_on = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                count = 0
                for cam_id in calibration_params.keys():
                    imgpt = voxel_lut[cam_id][(x, y, z)]
                    mask = foreground_masks[cam_id]
                    h, w = mask.shape
                    u, v = imgpt
                    if u < 0 or u >= w or v < 0 or v >= h:
                        continue
                    if mask[v, u] > 0:
                        count += 1
                if count >= min_views:
                    voxels_on.append([x, y, z])
    return voxels_on

def main():
    cam_ids = ['cam1', 'cam2', 'cam3', 'cam4']
    base_path = os.path.join("Assignment 2", "data")
    calibration_params = {}
    foreground_masks = {}
    
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
            print(f"Extrinsics not found for {cam}. Please ensure XML contains rvec and tvec.")
            continue
        calibration_params[cam] = (mtx, dist, rvec, tvec)
        
        background_video = os.path.join(cam_path, "background.avi")
        try:
            bg_model = build_background_model(background_video, num_frames=30)
        except Exception as e:
            print(f"Error building background model for {cam}: {e}")
            continue
        
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

    voxels_on = voxel_reconstruction(foreground_masks, calibration_params, voxel_step=8, grid_dims=(128, 64, 128), min_views=3)
    print(f"Number of voxels reconstructed: {len(voxels_on)}")
    
if __name__ == "__main__":
    main()
