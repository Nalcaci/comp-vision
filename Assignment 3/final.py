import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

# ----------------------
# Custom CIFAR-100 Wrapper to Return Coarse Labels
# ----------------------
class CIFAR100Coarse(torchvision.datasets.CIFAR100):
    """
    A wrapper around torchvision.datasets.CIFAR100 that converts fine labels
    into coarse labels (20 classes) using a fixed mapping.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root=root, train=train, transform=transform, 
                         target_transform=target_transform, download=download)
        # Hard-coded mapping from fine labels (0-99) to coarse labels (0-19)
        self.fine_to_coarse = [
            4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
            3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
            6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
            0, 11, 1, 10, 12, 16, 12, 1, 9, 15,
            13, 16, 16, 2, 4, 2, 10, 0, 17, 3,
            12, 9, 6, 4, 17, 0, 17, 5, 19, 2,
            5, 19, 2, 0, 1, 1, 4, 6, 3, 16,
            12, 9, 13, 16, 16, 13, 16, 19, 2, 5,
            4, 13, 0, 14, 14, 7, 5, 18, 3, 9,
            18, 2, 13, 14, 5, 7, 18, 4, 18, 19
        ]
    
    def __getitem__(self, index):
        # Get image and the original fine label from the parent class
        img, fine_label = super().__getitem__(index)
        coarse_label = self.fine_to_coarse[fine_label]
        return img, coarse_label

# ----------------------
# Data Loading Functions
# ----------------------
def load_cifar10(batch_size=32, split_ratio=0.8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_size = int(split_ratio * len(trainset))
    val_size   = len(trainset) - train_size
    train_subset, val_subset = random_split(trainset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def load_cifar100(batch_size=32, split_ratio=0.8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    # Use the custom wrapper to get coarse labels (20 classes)
    trainset = CIFAR100Coarse(root='./data', train=True, download=True, transform=transform)
    testset  = CIFAR100Coarse(root='./data', train=False, download=True, transform=transform)
    train_size = int(split_ratio * len(trainset))
    val_size   = len(trainset) - train_size
    train_subset, val_subset = random_split(trainset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# ----------------------
# Model Definitions
# ----------------------
class LeNet5(nn.Module):
    """
    LeNet-5 implementation with options for:
      - num_classes: number of output classes (default 10 for CIFAR-10)
      - pooling_type: 'avg' for average pooling or 'max' for max pooling
      - dropout: if True, applies dropout after fc1 (p=0.5)
    """
    def __init__(self, num_classes=10, pooling_type='avg', dropout=False):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)  # 32x32 -> 28x28
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # 28x28 -> 10x10 after pooling
        if pooling_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError("pooling_type must be either 'avg' or 'max'")
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.use_dropout = dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.25)
        
        # Weight initialization with Kaiming Uniform
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(x.size(0), -1)             # Flatten feature maps
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def modify_final_layer(model, new_num_classes):
    """Replace the final fully connected layer with a new one for a different number of classes."""
    in_features = model.fc3.in_features
    model.fc3 = nn.Linear(in_features, new_num_classes)
    nn.init.kaiming_uniform_(model.fc3.weight, nonlinearity='relu')
    return model

# ----------------------
# Training and Evaluation Functions
# ----------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return running_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return running_loss / len(loader), correct / total

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")
    return history

def evaluate_model(model, loader, device):
    """Evaluate model on given loader and return accuracy, predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total, all_preds, all_labels

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()
    
def plot_loss_curves(history_baseline, history_variant1, history_variant2):
    epochs = range(1, len(history_baseline["epoch"]) + 1)

    plt.figure(figsize=(10, 6))

    # Plot baseline LeNet model
    plt.plot(epochs, history_baseline["train_loss"], label="Train Loss (LeNet)", linestyle='-', marker='o')
    plt.plot(epochs, history_baseline["val_loss"], label="Val Loss (LeNet)", linestyle='--', marker='x')

    # Plot variant 1 model
    plt.plot(epochs, history_variant1["train_loss"], label="Train Loss (Variant 1)", linestyle='-', marker='o')
    plt.plot(epochs, history_variant1["val_loss"], label="Val Loss (Variant 1)", linestyle='--', marker='x')

    # Plot variant 2 model
    plt.plot(epochs, history_variant2["train_loss"], label="Train Loss (Variant 2)", linestyle='-', marker='o')
    plt.plot(epochs, history_variant2["val_loss"], label="Val Loss (Variant 2)", linestyle='--', marker='x')

    # Adding labels and title
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves for CIFAR Models")
    plt.legend()

    # Show the plot
    plt.show()

# ----------------------
# Main Experiment Routine
# ----------------------
def main():
    # Hyperparameters and device settings
    epochs = 10
    batch_size = 32
    initial_lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("saved_models", exist_ok=True)
    
    print("==== Loading CIFAR-10 Data ====")
    cifar10_train, cifar10_val, cifar10_test = load_cifar10(batch_size=batch_size, split_ratio=0.8)
    
    # ----------------------
    # Train three CIFAR-10 Models
    # ----------------------
    print("\n==== Training CIFAR-10 Models ====")
    criterion = nn.CrossEntropyLoss()
    
    # Baseline LeNet-5 (using Average Pooling)
    model_baseline = LeNet5(num_classes=10, pooling_type='avg', dropout=False).to(device)
    optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=initial_lr)
    print("\nTraining Baseline LeNet-5 (AvgPool)...")
    history_baseline = train_and_validate(model_baseline, cifar10_train, cifar10_val,
                                          criterion, optimizer_baseline, epochs, device)
    torch.save(model_baseline.state_dict(), "saved_models/cifar10_baseline.pth")
    
    # Variant 1: Add dropout after fc1 (still AvgPool)
    model_variant1 = LeNet5(num_classes=10, pooling_type='avg', dropout=True).to(device)
    optimizer_variant1 = optim.Adam(model_variant1.parameters(), lr=initial_lr)
    print("\nTraining Variant 1 (with Dropout)...")
    history_variant1 = train_and_validate(model_variant1, cifar10_train, cifar10_val,
                                          criterion, optimizer_variant1, epochs, device)
    torch.save(model_variant1.state_dict(), "saved_models/cifar10_variant1.pth")
    
    # Variant 2: Replace AvgPool with MaxPool
    model_variant2 = LeNet5(num_classes=10, pooling_type='max', dropout=False).to(device)
    optimizer_variant2 = optim.Adam(model_variant2.parameters(), lr=initial_lr)
    print("\nTraining Variant 2 (MaxPool instead of AvgPool)...")
    history_variant2 = train_and_validate(model_variant2, cifar10_train, cifar10_val,
                                          criterion, optimizer_variant2, epochs, device)
    torch.save(model_variant2.state_dict(), "saved_models/cifar10_variant2.pth")
    
    # ----------------------
    # Performance Summary for CIFAR-10 Models
    # ----------------------
    summary = []
    for name, hist in zip(["Baseline", "Variant1 (Dropout)", "Variant2 (MaxPool)"],
                           [history_baseline, history_variant1, history_variant2]):
        summary.append({
            "Model": name,
            "Final Train Loss": hist["train_loss"][-1],
            "Final Train Acc (%)": hist["train_acc"][-1]*100,
            "Final Val Loss": hist["val_loss"][-1],
            "Final Val Acc (%)": hist["val_acc"][-1]*100,
        })
    df_summary = pd.DataFrame(summary)
    print("\n==== CIFAR-10 Models Performance Summary ====")
    print(df_summary)
    
    # Decide on best model based on final validation accuracy
    final_accs = [history_baseline["val_acc"][-1], history_variant1["val_acc"][-1], history_variant2["val_acc"][-1]]
    best_idx = final_accs.index(max(final_accs))
    best_model_name = ["Baseline", "Variant1 (Dropout)", "Variant2 (MaxPool)"][best_idx]
    if best_idx == 0:
        best_model = model_baseline
    elif best_idx == 1:
        best_model = model_variant1
    else:
        best_model = model_variant2
    print(f"\nSelected Best CIFAR-10 Model: {best_model_name}")
    torch.save(best_model.state_dict(), "saved_models/best_cifar10_model.pth")
    
    # ----------------------
    # Train CIFAR-100 Model Using Best Architecture
    # ----------------------
    print("\n==== Training CIFAR-100 Model ====")
    cifar100_train, cifar100_val, cifar100_test = load_cifar100(batch_size=batch_size, split_ratio=0.8)
    # Instantiate best architecture (assumed Variant2: MaxPool, no dropout) for 20 classes
    model_cifar100 = LeNet5(num_classes=20, pooling_type='max', dropout=False).to(device)
    optimizer_cifar100 = optim.Adam(model_cifar100.parameters(), lr=initial_lr)
    history_cifar100 = train_and_validate(model_cifar100, cifar100_train, cifar100_val,
                                         criterion, optimizer_cifar100, epochs, device)
    torch.save(model_cifar100.state_dict(), "saved_models/cifar100_model.pth")
    
    # ----------------------
    # Fine-tune Pretrained CIFAR-100 Model on CIFAR-10
    # ----------------------
    print("\n==== Fine-Tuning CIFAR-10 Pretrained Model ====")
    cifar10_train_ft, cifar10_val_ft, cifar10_test_ft = load_cifar10(batch_size=batch_size, split_ratio=0.8)
    # Load the pretrained CIFAR-100 model and modify final layer for 10 outputs
    model_finetune = LeNet5(num_classes=20, pooling_type='max', dropout=False).to(device)
    model_finetune.load_state_dict(torch.load("saved_models/cifar100_model.pth"))
    model_finetune = modify_final_layer(model_finetune, new_num_classes=10)
    # Move the modified model to the device (fix for mismatched device error)
    model_finetune = model_finetune.to(device)
    
    finetune_lr = initial_lr / 2
    optimizer_finetune = optim.Adam(model_finetune.parameters(), lr=finetune_lr)
    print(f"Fine-tuning with learning rate: {finetune_lr}")
    history_finetune = train_and_validate(model_finetune, cifar10_train_ft, cifar10_val_ft,
                                          criterion, optimizer_finetune, epochs, device)
    torch.save(model_finetune.state_dict(), "saved_models/cifar10_pretrained.pth")
    
    # ----------------------
    # Final Evaluation on CIFAR-10 Test Set
    # ----------------------
    print("\n==== Final Evaluation on CIFAR-10 Test Set ====")
    best_cifar10 = LeNet5(num_classes=10, pooling_type='max' if best_model_name=="Variant2 (MaxPool)" else 'avg',
                          dropout=(True if best_model_name=="Variant1 (Dropout)" else False)).to(device)
    best_cifar10.load_state_dict(torch.load("saved_models/best_cifar10_model.pth"))
    
    cifar10_pretrained = LeNet5(num_classes=10, pooling_type='max', dropout=False).to(device)
    cifar10_pretrained.load_state_dict(torch.load("saved_models/cifar10_pretrained.pth"))
    
    test_acc_best, preds_best, labels_best = evaluate_model(best_cifar10, cifar10_test, device)
    test_acc_pretrained, preds_pretrained, labels_pretrained = evaluate_model(cifar10_pretrained, cifar10_test, device)
    
    print(f"Best CIFAR-10 Model Test Accuracy: {test_acc_best*100:.2f}%")
    print(f"CIFAR10_pretrained Model Test Accuracy: {test_acc_pretrained*100:.2f}%")
    
    plot_loss_curves(history_baseline, history_variant1, history_variant2)
    
    # Compute confusion matrices
    cm_best = confusion_matrix(labels_best, preds_best)
    cm_pretrained = confusion_matrix(labels_pretrained, preds_pretrained)
    classes = cifar10_test.dataset.classes
    
    # Plot confusion matrices
    plot_confusion_matrix(cm_best, classes, title="Best CIFAR-10 Model Confusion Matrix")
    plot_confusion_matrix(cm_pretrained, classes, title="CIFAR10_pretrained Model Confusion Matrix")
    
    # ----------------------
    # Comparison Explanation (printed)
    # ----------------------
    print("\n==== Comparison and Explanation ====")
    print("1. Training Sets:")
    print("   - The Best CIFAR-10 model was trained solely on CIFAR-10 using a split of training and validation data.")
    print("   - The CIFAR10_pretrained model was first trained on CIFAR-100 (with 20 coarse labels) and then fine-tuned on CIFAR-10.")
    print("2. Validation Sets:")
    print("   - Both models used a portion of the CIFAR-10 training data as validation sets for hyperparameter tuning.")
    print("3. Test Sets:")
    print("   - The CIFAR-10 test set was held out during training and validation, providing an unbiased evaluation of generalization.")
    print("4. Performance Differences:")
    print("   - The best model (trained directly on CIFAR-10) may capture class-specific features more directly.")
    print("   - The fine-tuned model benefits from pretraining on CIFAR-100, which may lead to differences in learned features.")

if __name__ == '__main__':
    main()
