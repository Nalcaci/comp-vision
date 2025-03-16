import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Model Definitions ---
# Best CIFAR-10 model architecture (assumed to be the one with max pooling variant)
class LeNet5Variant2(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5Variant2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)   # 32x32 -> 28x28
        self.pool1 = nn.MaxPool2d(2, 2)     # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(6, 16, 5)    # 14x14 -> 10x10
        self.pool2 = nn.MaxPool2d(2, 2)     # 10x10 -> 5x5
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
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# CIFAR10_pretrained model architecture (same as the model used for fine-tuning)
class LeNet5Variant2_Mod(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5Variant2_Mod, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Utility Functions ---
def evaluate_model(model, loader, device):
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
    accuracy = correct / total
    return accuracy, all_preds, all_labels

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()

# --- Main Evaluation Routine ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data transformation and loading for CIFAR-10 test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    classes = testset.classes
    
    # Load the saved models
    best_model_path = "best_cifar10_model.pth"
    pretrained_model_path = "cifar10_pretrained.pth"
    
    best_model = LeNet5Variant2(num_classes=10).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    
    cifar10_pretrained = LeNet5Variant2_Mod(num_classes=10).to(device)
    cifar10_pretrained.load_state_dict(torch.load(pretrained_model_path))
    
    # Evaluate both models on the test set
    best_acc, best_preds, best_labels = evaluate_model(best_model, test_loader, device)
    pretrained_acc, pretrained_preds, pretrained_labels = evaluate_model(cifar10_pretrained, test_loader, device)
    
    print(f"Best CIFAR-10 Model Test Accuracy: {best_acc*100:.2f}%")
    print(f"CIFAR10_pretrained Model Test Accuracy: {pretrained_acc*100:.2f}%\n")
    
    # Compute confusion matrices
    cm_best = confusion_matrix(best_labels, best_preds)
    cm_pretrained = confusion_matrix(pretrained_labels, pretrained_preds)
    
    print("Confusion Matrix for Best CIFAR-10 Model:")
    print(cm_best)
    print("\nConfusion Matrix for CIFAR10_pretrained Model:")
    print(cm_pretrained)
    
    # Plot confusion matrices
    plot_confusion_matrix(cm_best, classes, title="Best CIFAR-10 Model Confusion Matrix")
    plot_confusion_matrix(cm_pretrained, classes, title="CIFAR10_pretrained Model Confusion Matrix")
    
    # --- Comparison Explanation ---
    print("\n--- Comparison and Explanation ---")
    print("1. **Training Sets:**")
    print("   - The Best CIFAR-10 model was trained solely on CIFAR-10 data. Its training set was used to directly learn features specific to CIFAR-10, while the validation set guided hyperparameter tuning.")
    print("   - The CIFAR10_pretrained model was initially trained on CIFAR-100 and then fine-tuned on CIFAR-10. Its pretraining may have exposed it to more diverse features, though the fine-tuning adjusts it for CIFAR-10.")
    print("\n2. **Validation Sets:**")
    print("   - Both models used a portion of the CIFAR-10 training data as a validation set for early stopping and hyperparameter adjustments. This set helps monitor overfitting during training.")
    print("\n3. **Test Sets:**")
    print("   - The test set in CIFAR-10 is completely held out during training and validation. It represents unseen data and provides an unbiased estimate of model generalization.")
    print("\n4. **Performance Differences:**")
    print("   - The Best CIFAR-10 model is expected to perform well since it was trained directly on CIFAR-10.")
    print("   - The CIFAR10_pretrained model, although fine-tuned, might show slight differences due to its pretraining on CIFAR-100. Its generalization may be different if the pretraining provided beneficial or adverse feature representations.")
    
if __name__ == '__main__':
    main()
