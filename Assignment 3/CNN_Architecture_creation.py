import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Define the LeNet-5 model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)  # 32x32x3 -> 28x28x6
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)                      # 28x28x6 -> 14x14x6
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # 14x14x6 -> 10x10x16
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)                      # 10x10x16 -> 5x5x16
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5x16 -> 120
        self.fc2 = nn.Linear(120, 84)         # 120 -> 84
        self.fc3 = nn.Linear(84, 10)          # 84 -> 10 (Output)

        # Weight initialization using Kaiming Uniform
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = torch.flatten(x, 1)               # Flatten feature maps
        x = F.relu(self.fc1(x))               # Fully connected 1 -> ReLU
        x = F.relu(self.fc2(x))               # Fully connected 2 -> ReLU
        x = self.fc3(x)                       # Fully connected 3 (output)
        return x

# Define hyperparameters
batch_size = 32
learning_rate = 0.001

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split into training and validation (80/20)
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Instantiate model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)

criterion = nn.CrossEntropyLoss()  # Softmax is included in CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Print model summary
print(model)
