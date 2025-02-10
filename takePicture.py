import cv2
import os

# Ensure "images" folder exists
if not os.path.exists('images'):
    os.makedirs('images')

cap = cv2.VideoCapture(0)  # Open the webcam

image_count = len(os.listdir('images'))  # Count existing images to avoid overwriting

while True:
    ret, frame = cap.read()  # Continuously capture frames
    if not ret:
        print("Failed to capture image")
        break

    cv2.imshow('Live Video', frame)  # Show live video feed

    key = cv2.waitKey(1) & 0xFF  # Wait for key press
    if key == ord('y'):  # Save image when 'y' is pressed
        image_count += 1
        filename = f'images/c{image_count}.png'
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")
    elif key == ord('q'):  # Quit when 'q' is pressed
        break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close OpenCV windows
