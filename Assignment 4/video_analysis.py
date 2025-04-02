import cv2
import torch
import numpy as np
from model import YourModel  # Replace with your actual model import
from torchvision import transforms

# Load trained model
model = YourModel()  # Replace with actual model class
model.load_state_dict(torch.load("saved_models/yolo_model"))  # Load weights
model.eval()

# Video settings
input_video_path = "data/c_d_video.mp4"
output_video_path = "data/c_d_video_output.mp4"

# Open video
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Resize to match model input size
])

# Process video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Run model
    with torch.no_grad():
        predictions = model(img_tensor)

    # Parse predictions (assuming [x1, y1, x2, y2, class])
    for pred in predictions:
        x1, y1, x2, y2, cls = map(int, pred.tolist())  # Convert to int
        color = (0, 0, 255) if cls == 0 else (255, 0, 0)  # Red for cat, blue for dog
        label = "Cat" if cls == 0 else "Dog"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write frame to output video
    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved as {output_video_path}")
