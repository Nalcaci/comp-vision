import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

# --- Modified Model for CIFAR-100 with 20 Classes ---
# Here we adapt the best architecture (assumed as the max-pooling variant)
# by changing the final output layer from 10 to 20 outputs.
class LeNet5Variant2_Mod(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5Variant2_Mod, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # Change the final layer to output `num_classes` (set to 20 for CIFAR-100_model)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Weight initialization using Kaiming Uniform
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
    # Store loss and accuracy for each epoch
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}/{epochs} => Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    return history

# --- Main Routine for CIFAR-100_model Training ---
def main():
    # Hyperparameters remain the same as before
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    # Load CIFAR-100 dataset
    full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    # Split CIFAR-100 training set into training and validation sets (80/20)
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    train_subset, val_subset = random_split(full_trainset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # Instantiate the best architecture from CIFAR-10 experiments, modified to output 20 classes.
    model = LeNet5Variant2_Mod(num_classes=20).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training CIFAR100_model (Best CIFAR-10 architecture adapted for 20 outputs)...")
    history = train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, device)
    
    # Final evaluation on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc*100:.2f}%")
    
if __name__ == '__main__':
    main()
