import os
import random
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ---------------------
# Annotation Parsing
# ---------------------
def parse_annotation(xml_file):
    """
    Parse a single XML file and return a dictionary with:
    filename, label (0 for cat, 1 for dog), x, y, w, h.
    Assumes one object per image.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image filename
    filename = root.find('filename').text

    # Get the first object annotation (assumes one object per image)
    obj = root.find('object')
    if obj is None:
        return None

    # Get label (assumes label is "cat" or "dog")
    label_str = obj.find('name').text.lower()
    if label_str == 'cat':
        label = 0
    elif label_str == 'dog':
        label = 1
    else:
        return None  # Skip if label is not cat or dog

    # Get bounding box (assumes VOC format: xmin, ymin, xmax, ymax)
    bndbox = obj.find('bndbox')
    xmin = float(bndbox.find('xmin').text)
    ymin = float(bndbox.find('ymin').text)
    xmax = float(bndbox.find('xmax').text)
    ymax = float(bndbox.find('ymax').text)
    w = xmax - xmin
    h = ymax - ymin

    return {"filename": filename, "label": label, "x": xmin, "y": ymin, "w": w, "h": h}

def load_annotations(annotations_folder):
    """
    Parse all XML files in the annotations folder (including subfolders) and return a DataFrame.
    Searches recursively for files with extension .xml or .XML.
    """
    data = []
    # Recursively search for XML files in the folder.
    xml_files = glob.glob(os.path.join(annotations_folder, '**', '*.xml'), recursive=True)
    xml_files += glob.glob(os.path.join(annotations_folder, '**', '*.XML'), recursive=True)
    print(f"Found {len(xml_files)} XML files in {annotations_folder}")
    
    for xml_file in xml_files:
        annotation = parse_annotation(xml_file)
        if annotation is not None:
            data.append(annotation)
    return pd.DataFrame(data)

# ---------------------
# Utility functions
# ---------------------
def cell_to_bbox(cell_row, cell_col, pred):
    """
    Convert a cell’s prediction (x, y, w, h) into an absolute bounding box.
    x and y are offsets within the cell (0-1), and w, h are normalized by the full image size.
    """
    cell_size = 112 / 7  # each cell covers ~16 pixels
    x_cell, y_cell, w, h = pred
    # Compute the absolute center coordinates
    center_x = (cell_col + x_cell) * cell_size
    center_y = (cell_row + y_cell) * cell_size
    box_w = w * 112
    box_h = h * 112
    # Convert center coordinates to corner coordinates
    x1 = center_x - box_w / 2
    y1 = center_y - box_h / 2
    x2 = center_x + box_w / 2
    y2 = center_y + box_h / 2
    return [x1, y1, x2, y2]

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

# ---------------------
# Dataset definition
# ---------------------
class DogCatDataset(Dataset):
    """
    A PyTorch Dataset for the dog-cat head detection dataset.
    Uses a dataframe with columns: filename, label, x, y, w, h.
    """
    def __init__(self, df, images_folder, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_folder = images_folder
        self.transform = transform
        # Always resize images to 112x112.
        self.resize = transforms.Resize((112, 112))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_folder, row["filename"])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Bounding box in pixel coordinates from XML: x, y, w, h (x,y = top-left)
        bbox = np.array([row["x"], row["y"], row["w"], row["h"]], dtype=np.float32)

        # Resize image and adjust bbox accordingly
        image = self.resize(image)
        new_w, new_h = 112, 112
        scale_w = new_w / orig_w
        scale_h = new_h / orig_h
        bbox[0] *= scale_w
        bbox[1] *= scale_h
        bbox[2] *= scale_w
        bbox[3] *= scale_h

        image = self.to_tensor(image)

        # Prepare target tensor in YOLO format.
        # Shape: (7, 7, 7) where last dim: [x, y, w, h, objectness, class0, class1]
        target = np.zeros((7, 7, 7), dtype=np.float32)
        # Compute normalized center of the bounding box
        center_x = (bbox[0] + bbox[2] / 2) / 112  
        center_y = (bbox[1] + bbox[3] / 2) / 112  
        cell_col = int(center_x * 7)
        cell_row = int(center_y * 7)
        # Compute offsets relative to the cell
        x_cell = center_x * 7 - cell_col
        y_cell = center_y * 7 - cell_row
        norm_w = bbox[2] / 112
        norm_h = bbox[3] / 112
        target[cell_row, cell_col, 0:4] = [x_cell, y_cell, norm_w, norm_h]
        target[cell_row, cell_col, 4] = 1  # objectness
        
        # One-hot encoding for class; label: 0=cat, 1=dog.
        label = int(row["label"])
        if label == 0:
            target[cell_row, cell_col, 5] = 1
        else:
            target[cell_row, cell_col, 6] = 1

        # Flatten target to shape (343,)
        target = torch.from_numpy(target.flatten())
        return image, target

# ---------------------
# Model definition
# ---------------------
class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7 * 7 * 32, 512)
        self.fc2 = nn.Linear(512, 343)  # 7x7 grid * 7 values per cell
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# ---------------------
# Loss definition
# ---------------------
class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=1, C=2, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        # Reshape predictions and target to [batch, 7, 7, 5+C]
        predictions = predictions.view(-1, self.S, self.S, 5 + self.C)
        target = target.view(-1, self.S, self.S, 5 + self.C)
        
        # Create object and no-object masks
        obj_mask = target[..., 4] > 0.5
        noobj_mask = ~obj_mask

        # Coordinate loss for x and y (apply mask after slicing the last dim)
        coord_loss = self.lambda_coord * (
            (predictions[..., 0:2][obj_mask] - target[..., 0:2][obj_mask]) ** 2
        ).sum()

        # Width and height loss (apply sqrt to lessen the impact of large values)
        pred_wh = torch.sqrt(predictions[..., 2:4][obj_mask] + 1e-6)
        target_wh = torch.sqrt(target[..., 2:4][obj_mask] + 1e-6)
        wh_loss = self.lambda_coord * ((pred_wh - target_wh) ** 2).sum()

        # Confidence loss
        conf_loss_obj = ((predictions[..., 4][obj_mask] - target[..., 4][obj_mask]) ** 2).sum()
        conf_loss_noobj = self.lambda_noobj * ((predictions[..., 4][noobj_mask] - target[..., 4][noobj_mask]) ** 2).sum()

        # Classification loss (only for cells with objects)
        class_loss = ((predictions[..., 5:][obj_mask] - target[..., 5:][obj_mask]) ** 2).sum()

        total_loss = coord_loss + wh_loss + conf_loss_obj + conf_loss_noobj + class_loss
        return total_loss

# ---------------------
# Training loop
# ---------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_wts = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    return model

# ---------------------
# Evaluation functions
# ---------------------
def compute_average_precision(all_detections, all_ground_truths, class_idx, iou_threshold=0.5):
    scores = []
    tp = []
    num_gt = 0
    for dets, gts in zip(all_detections, all_ground_truths):
        gt_boxes = [gt[0] for gt in gts if gt[1] == class_idx]
        num_gt += len(gt_boxes)
        used = [False] * len(gt_boxes)
        for box, cls, conf in sorted(dets, key=lambda x: x[2], reverse=True):
            if cls != class_idx:
                continue
            scores.append(conf)
            iou_max = 0
            match_idx = -1
            for i, gt_box in enumerate(gt_boxes):
                iou = compute_iou(box, gt_box)
                if iou > iou_threshold and iou > iou_max and not used[i]:
                    iou_max = iou
                    match_idx = i
            if match_idx >= 0:
                tp.append(1)
                used[match_idx] = True
            else:
                tp.append(0)
    if num_gt == 0:
        return 0
    tp = np.array(tp)
    scores = np.array(scores)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall = cum_tp / (num_gt + 1e-6)
    ap = np.trapz(precision, recall)
    return ap

def evaluate_model(model, dataloader, threshold, device):
    model.eval()
    all_detections = []
    all_ground_truths = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(-1, 7, 7, 7)
            targets = targets.view(-1, 7, 7, 7)
            batch_size = images.size(0)
            for b in range(batch_size):
                pred = outputs[b].cpu().numpy()
                target_np = targets[b].cpu().numpy()
                image_detections = []
                image_ground_truth = []
                for i in range(7):
                    for j in range(7):
                        if target_np[i, j, 4] > 0.5:
                            gt_box = cell_to_bbox(i, j, target_np[i, j, 0:4])
                            gt_class = np.argmax(target_np[i, j, 5:])
                            image_ground_truth.append((gt_box, gt_class))
                        if pred[i, j, 4] > threshold:
                            pred_box = cell_to_bbox(i, j, pred[i, j, 0:4])
                            pred_conf = pred[i, j, 4]
                            pred_class = np.argmax(pred[i, j, 5:])
                            image_detections.append((pred_box, pred_class, pred_conf))
                all_detections.append(image_detections)
                all_ground_truths.append(image_ground_truth)
    ap_cat = compute_average_precision(all_detections, all_ground_truths, class_idx=0)
    ap_dog = compute_average_precision(all_detections, all_ground_truths, class_idx=1)
    mAP = (ap_cat + ap_dog) / 2
    return mAP

def compute_confusion_matrix(model, dataloader, threshold, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(-1, 7, 7, 7)
            targets = targets.view(-1, 7, 7, 7)
            batch_size = images.size(0)
            for b in range(batch_size):
                target_np = targets[b].cpu().numpy()
                gt_class = None
                for i in range(7):
                    for j in range(7):
                        if target_np[i, j, 4] > 0.5:
                            gt_class = np.argmax(target_np[i, j, 5:])
                            break
                    if gt_class is not None:
                        break
                pred = outputs[b].cpu().numpy()
                best_conf = -1
                pred_class = None
                for i in range(7):
                    for j in range(7):
                        if pred[i, j, 4] > threshold and pred[i, j, 4] > best_conf:
                            best_conf = pred[i, j, 4]
                            pred_class = np.argmax(pred[i, j, 5:])
                if gt_class is not None:
                    y_true.append(gt_class)
                    y_pred.append(pred_class if pred_class is not None else -1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return cm

# ---------------------
# Visualization helper
# ---------------------
def visualize_samples(dataset, num_samples=5):
    indices = random.sample(range(len(dataset)), num_samples)
    for idx in indices:
        image, target = dataset[idx]
        image_np = image.permute(1, 2, 0).numpy()
        target = target.view(7, 7, 7)
        fig, ax = plt.subplots(1)
        ax.imshow(image_np)
        for i in range(7):
            for j in range(7):
                if target[i, j, 4] > 0.5:
                    box = cell_to_bbox(i, j, target[i, j, 0:4].numpy())
                    rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                         fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
                    cls = int(torch.argmax(target[i, j, 5:]).item())
                    label_str = "cat" if cls == 0 else "dog"
                    ax.text(box[0], box[1], label_str, color='yellow', fontsize=12)
        plt.show()

# ---------------------
# Main function
# ---------------------
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load annotations from XML files
    annotations_folder = "data/annotations"
    df = load_annotations(annotations_folder)
    print(f"Loaded {len(df)} annotations.")

    # Stratified split on label
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Folder containing images
    images_folder = "data/Images"

    # Create datasets
    train_dataset = DogCatDataset(train_df, images_folder)
    val_dataset = DogCatDataset(val_df, images_folder)

    # Visualize a few samples to verify resizing and bbox transformation
    visualize_samples(train_dataset, num_samples=5)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model, loss, optimizer
    model = YOLOv1().to(device)
    criterion = YOLOLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)

    # Evaluate model over a range of thresholds
    thresholds = np.linspace(0, 1, 21)
    mAPs = []
    for thr in thresholds:
        mAP = evaluate_model(model, val_loader, threshold=thr, device=device)
        mAPs.append(mAP)
        print(f"Threshold {thr:.2f}: mAP = {mAP:.4f}")

    best_thr = thresholds[np.argmax(mAPs)]
    print(f"Best threshold based on mAP: {best_thr:.2f}")

    # Compute and display confusion matrix
    cm = compute_confusion_matrix(model, val_loader, threshold=best_thr, device=device)
    print("Confusion Matrix (rows: true, cols: predicted [cat, dog]):")
    print(cm)

    # Plot mAP vs threshold
    plt.figure()
    plt.plot(thresholds, mAPs, marker='o')
    plt.xlabel("Objectness Threshold")
    plt.ylabel("mAP")
    plt.title("mAP vs Objectness Threshold")
    plt.show()

if __name__ == '__main__':
    main()
