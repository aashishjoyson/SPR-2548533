# -*- coding: utf-8 -*-
"""AashishJoyson_2548533_DL_LAB2

Original file is located at
    https://colab.research.google.com/drive/1xnt1vVL6lWjEZTD6jlUXGGDjX0TMPNBl
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define the transformation: Converting to Tensor and Normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#Training Data
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                          shuffle=True)

# Test Data
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                         shuffle=False)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show images
print("Sanity Check - Visualizing a batch of data:")
imshow(torchvision.utils.make_grid(images[:8]))
print('Labels:', ' '.join(f'{classes[labels[j]]}' for j in range(8)))

# Define the Neural Network Architecture
class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 784 (28x28 flattened), Output: 10 classes
        # Funnel Architecture: 784 -> 256 -> 128 -> 64 -> 10
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        # Output layer
        x = self.fc4(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FashionClassifier().to(device)

print("Model Architecture Created:")
print(model)

# 1. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
train_accuracies = []

# 2. The Training Loop
num_epochs = 10
print(f"Starting Training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # B. Forward Pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # C. Backward Pass and Optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for this epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total


    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

print("Training Finished!")

# 1. Plot the Learning Curves
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='red')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy', color='blue')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.show()

print("\nStarting Evaluation on Test Data...")
model.eval()
correct = 0
total = 0

with torch.no_grad():# for savinfg speed and memeory
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"Final Test Accuracy: {test_acc:.2f}%")

print("-" * 30)
print(f"Training Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Test Accuracy:     {test_acc:.2f}%")
print("-" * 30)

if abs(train_accuracies[-1] - test_acc) < 5:
    print("Conclusion: The model generalizes well (No significant overfitting).")
else:
    print("Conclusion: There is a gap between Train and Test (Possible overfitting).")

# Advanced Task 2: Activation Function Study
class DynamicNet(nn.Module):
    def __init__(self, activation_name):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        if activation_name == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation_name == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        return x

# The Experiment
activations = ['relu', 'sigmoid', 'tanh']
history = {}

print("Starting Advanced Task 2: Activation Comparison...")

for act_name in activations:
    print(f"\n--- Training with {act_name.upper()} ---")

    model_var = DynamicNet(act_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_var.parameters(), lr=0.001)

    losses = []

    # 2. Short Training Run
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model_var(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    history[act_name] = losses

# 3. Visualization
plt.figure(figsize=(10, 6))
for act_name in activations:
    plt.plot(history[act_name], label=f'{act_name.upper()}')

plt.title('Comparison of Activation Functions (Convergence Speed)')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

def visualize_activations(model, image):
    model.eval()

    x = image.to(device).view(-1, 784)

    # 2. Pass through layers manually to capture intermediate outputs
    with torch.no_grad():
        h1 = model.relu(model.fc1(x))
        h2 = model.relu(model.fc2(h1))
        h3 = model.relu(model.fc3(h2))

    plt.figure(figsize=(15, 4))

    # Input Image
    plt.subplot(1, 4, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title("Input Image")
    plt.axis('off')

    # Layer 1 Activation (Reshaped to 16x16 grid for visualization)
    plt.subplot(1, 4, 2)
    plt.imshow(h1.cpu().view(16, 16), cmap='viridis')
    plt.title("Layer 1 (256 Neurons)")
    plt.axis('off')

    # Layer 2 Activation (Reshaped to 16x8 grid)
    plt.subplot(1, 4, 3)
    plt.imshow(h2.cpu().view(16, 8), cmap='viridis')
    plt.title("Layer 2 (128 Neurons)")
    plt.axis('off')

    # Layer 3 Activation (Reshaped to 8x8 grid)
    plt.subplot(1, 4, 4)
    plt.imshow(h3.cpu().view(8, 8), cmap='viridis')
    plt.title("Layer 3 (64 Neurons)")
    plt.axis('off')

    plt.show()


print("\nVisualizing Network 'Thoughts' ")
dataiter = iter(test_loader)
images, labels = next(dataiter)
sample_image = images[0]

visualize_activations(model, sample_image)

# Final Experiment: The Overfitting Stress Test
# We train for 20 epochs and track Test Loss to find the "tipping point"

print("Starting Overfitting Stress Test (20 Epochs)...")


model = FashionClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss = test_running_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total


    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")


plt.figure(figsize=(12, 5))

# Plot 1: Loss (The "Overfitting Check")
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss', color='blue')
plt.plot(history['test_loss'], label='Test (Validation) Loss', color='red', linestyle='--')
plt.title('Loss Curve: Train vs Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot 2: Accuracy
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Training Acc', color='blue')
plt.plot(history['test_acc'], label='Test Acc', color='red', linestyle='--')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.show()

"""**Conclusion**



In this lab, I built a Neural Network to classify images of clothing (Fashion MNIST). My goal was to see how different settings affect the model's ability to learn.

What I Observed:

The Model Worked: I used a "funnel" design (shrinking from 784 inputs down to 10 outputs). It worked really well, reaching 91% accuracy during training and 87% on the test data. This proves the model actually learned to recognize the clothes.

ReLU is the Best Choice: When I compared different activation functions, ReLU (Blue line) was much faster and better than the others. The Sigmoid function was very slow at the start because of the "Vanishing Gradient" problem, so it's not good for this kind of deep network.

How the Network "Sees": I visualized the hidden layers. You can clearly see that the first layer looks at edges and outlines (like the shape of a shoe), while the deeper layers look at more abstract patterns to make the final decision.

Why I Stopped at 10 Epochs: I ran a stress test for 20 epochs. The graph shows that after Epoch 12, the Test Error (Red line) started going UP, even though the Training Error kept going down. This is Overfitting. It means the model started memorizing instead of learning. So, stopping at 10 epochs was the correct decision to get the best results.

Final Result: The experiment showed that a simple 4-layer network using ReLU is efficient and accurate for this task, as long as we stop training before it starts overfitting.
"""

