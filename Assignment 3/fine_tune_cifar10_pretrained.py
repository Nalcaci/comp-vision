import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

# --- Model Definition (Same as used for CIFAR100_model) ---
class LeNet5Variant2_Mod(nn.Module):
    def __init__(self, num_classes=20):
        super(LeNet5Variant2_Mod, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Weight initialization
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

def fine_tune(model, train_loader, val_loader, criterion, optimizer, epochs, device):
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

# --- Main Fine-Tuning Routine ---
def main():
    # Hyperparameters
    initial_lr = 0.001
    fine_tune_lr = initial_lr / 2  # Fine-tuning learning rate (e.g., 0.0005)
    epochs = 10
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data transformations for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # Load the pretrained CIFAR100_model (assumed saved in 'cifar100_model.pth')
    # This model was originally built for 20 classes.
    pretrained_model = LeNet5Variant2_Mod(num_classes=20)
    pretrained_model.load_state_dict(torch.load("cifar100_model.pth"))
    pretrained_model.to(device)
    
    # Change the last layer from 20 outputs back to 10 outputs for CIFAR-10.
    in_features = pretrained_model.fc3.in_features
    pretrained_model.fc3 = nn.Linear(in_features, 10)
    nn.init.kaiming_uniform_(pretrained_model.fc3.weight, nonlinearity='relu')
    
    # This modified model is now our CIFAR10_pretrained model.
    model = pretrained_model
    
    # Set up loss function and optimizer with the fine-tuning learning rate.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)
    
    print("Fine-tuning CIFAR10_pretrained on CIFAR-10 with learning rate", fine_tune_lr)
    history = fine_tune(model, train_loader, val_loader, criterion, optimizer, epochs, device)
    
    # Evaluate the fine-tuned model on the test set.
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc*100:.2f}%")
    
if __name__ == '__main__':
    main()
