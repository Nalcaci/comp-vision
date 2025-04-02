import cv2
import torch
from torchvision import transforms
from main import YOLOv1 
from main import cell_to_bbox  # Import the function from your training script

# Load model (replace with your actual model class)
model = YOLOv1().eval()  # From your implementation
model.load_state_dict(torch.load("saved_models/yolo_model"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Video settings
input_video_path = "data/c_d_video2.mp4"
output_video_path = "data/c_d_video_output2.mp4"
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
])

# Process video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        predictions = model(img_tensor).view(7, 7, 7).cpu()

    # Parse predictions
    obj_mask = predictions[..., 4] > 0.65  # Optimal threshold
    if obj_mask.any():
        cell_row, cell_col = torch.where(obj_mask)
        pred = predictions[cell_row[0], cell_col[0]]
        x1, y1, x2, y2 = cell_to_bbox(cell_row[0], cell_col[0], pred[0:4])
        
        # Scale coordinates
        scale_x = frame_width / 112
        scale_y = frame_height / 112
        x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
        x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
        
        # Draw
        color = (0, 0, 255) if pred[5] > pred[6] else (255, 0, 0)
        label = "Cat" if pred[5] > pred[6] else "Dog"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)

cap.release()
out.release()
print(f"Output saved to {output_video_path}")