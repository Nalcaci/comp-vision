import cv2
import os

video_path = "Assignment 2/data/cam4/checkerboard.avi"
output_image_path = "Assignment 2/data/cam4/checkerboard.png"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

ret, frame1 = cap.read()  # Read frame 1 (not used)
ret, frame2 = cap.read()  # Read frame 2 (this is the one we save)

if not ret:
    print("Error: Could not read the second frame.")
else:
    cv2.imwrite(output_image_path, frame2)
    print(f"Saved second frame as {output_image_path}")

cap.release()
cv2.destroyAllWindows()