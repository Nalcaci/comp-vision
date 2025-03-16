import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import pandas as pd

# --- Model Definitions ---

# Baseline LeNet-5 (CIFAR10_lenet) using average pooling
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)   # 32x32 -> 28x28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)                      # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)     # 14x14 -> 10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)                      # 10x10 -> 5x5
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        # Weight initialization
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # CrossEntropyLoss will apply softmax internally
        return x

# Variant 1 (CIFAR10_model1): Add dropout after fc1.
class LeNet5Variant1(nn.Module):
    def __init__(self):
        super(LeNet5Variant1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout(p=0.5)  # Added dropout for regularization
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Variant 2 (CIFAR10_model2): Replace average pooling with max pooling.
class LeNet5Variant2(nn.Module):
    def __init__(self):
        super(LeNet5Variant2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Changed to max pooling
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Changed to max pooling
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Training and Evaluation Functions ---

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
    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

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
    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    # To store epoch-wise metrics
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}/{epochs} => Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    return history

# --- Main Training Routine ---

def main():
    # Settings
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data transformations and loading for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create training/validation split (80/20)
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    train_subset, val_subset = random_split(full_trainset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # Define loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Dictionary to store performance histories for each model
    all_histories = {}
    
    # --- Train CIFAR10_lenet (Baseline) ---
    print("Training CIFAR10_lenet (Baseline LeNet-5 with AvgPool)...")
    model_lenet = LeNet5().to(device)
    optimizer = optim.Adam(model_lenet.parameters(), lr=learning_rate)
    history_lenet = train_and_validate(model_lenet, train_loader, val_loader, criterion, optimizer, epochs, device)
    all_histories["CIFAR10_lenet"] = history_lenet
    
    # --- Train CIFAR10_model1 (Variant with Dropout) ---
    print("\nTraining CIFAR10_model1 (LeNet-5 with Dropout)...")
    model_variant1 = LeNet5Variant1().to(device)
    optimizer = optim.Adam(model_variant1.parameters(), lr=learning_rate)
    history_variant1 = train_and_validate(model_variant1, train_loader, val_loader, criterion, optimizer, epochs, device)
    all_histories["CIFAR10_model1"] = history_variant1
    
    # --- Train CIFAR10_model2 (Variant with MaxPool) ---
    print("\nTraining CIFAR10_model2 (LeNet-5 with MaxPool instead of AvgPool)...")
    model_variant2 = LeNet5Variant2().to(device)
    optimizer = optim.Adam(model_variant2.parameters(), lr=learning_rate)
    history_variant2 = train_and_validate(model_variant2, train_loader, val_loader, criterion, optimizer, epochs, device)
    all_histories["CIFAR10_model2"] = history_variant2
    
    # --- Summarize Final Performance ---
    summary = []
    for model_name, hist in all_histories.items():
        # Get metrics from the final epoch
        final_epoch = -1
        summary.append({
            "Model": model_name,
            "Final Train Loss": hist["train_loss"][final_epoch],
            "Final Train Acc (%)": hist["train_acc"][final_epoch] * 100,
            "Final Val Loss": hist["val_loss"][final_epoch],
            "Final Val Acc (%)": hist["val_acc"][final_epoch] * 100,
        })
    df = pd.DataFrame(summary)
    print("\nFinal Performance Summary:")
    print(df)
    
if __name__ == '__main__':
    main()
