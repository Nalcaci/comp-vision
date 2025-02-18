import cv2
import os

# Load the video file
video_path = "Assignment 2/data/cam4/intrinsics.avi"  # Change this to the actual video file name in your project folder
cap = cv2.VideoCapture(video_path)

# Define the screenshot save directory
save_dir = "Assignment 2/data/cam4/intrinsics_screenshots"
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

paused = False  # Flag to track pause state
frame_count = 0  # Counter for saved frames

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        
        cv2.imshow("Video Player", frame)

    key = cv2.waitKey(30) & 0xFF  # Capture key press

    if key == ord('p'):  # Pause
        paused = True

    elif key == ord('c'):  # Continue playing
        paused = False

    elif key == ord('s') and paused:  # Save frame only if paused
        frame_filename = os.path.join(save_dir, f"paused_frame_{frame_count}.png")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame: {frame_filename}")
        frame_count += 1

    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
