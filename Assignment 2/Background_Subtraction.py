import cv2 as cv
import numpy as np
import glob

def get_background_frame(video_path, method="median", sample_rate=10):
    cap = cv.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    for i in range(0, total_frames, sample_rate):  # Sample frames to reduce computation
        cap.set(cv.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

    cap.release()

    if method == "median":
        background_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    else:
        background_frame = np.mean(frames, axis=0).astype(dtype=np.uint8)

    return background_frame

def background_subtraction(background_video, people_video, output_video):
    background = get_background_frame(background_video, method="median")
    
    cap = cv.VideoCapture(people_video)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    out = cv.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame = cv.GaussianBlur(gray_frame, (1, 1), 0)
        
        # Compute absolute difference
        diff = cv.absdiff(background, gray_frame)
        _, mask = cv.threshold(diff, 16, 255, cv.THRESH_BINARY)

        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)

        # Extract foreground
        foreground = cv.bitwise_and(frame, frame, mask=mask)

        out.write(foreground)
        cv.imshow("Foreground", foreground)

        if cv.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    base_path = "Assignment 2/data/"
    
    for cam_id in range(1, 5):
        bg_video = f"{base_path}cam{cam_id}/background.avi"
        people_video = f"{base_path}cam{cam_id}/video.avi"
        output_video = f"{base_path}cam{cam_id}/foreground_output.mp4"
        
        background_subtraction(bg_video, people_video, output_video)
        print(f"Processed background subtraction for camera {cam_id}")
