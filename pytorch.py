import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# XOR dataset
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=torch.float32)
y = torch.tensor([[0.],[1.],[1.],[0.]], dtype=torch.float32)

# Hyperparameters
learning_rate = 0.5
hidden_neurons = 8
epochs = 10000

# Model layers
W1 = nn.Linear(2, hidden_neurons)
W2 = nn.Linear(hidden_neurons, 1)
activation = nn.Tanh()

# Optimizer and loss
optimizer = optim.SGD(list(W1.parameters()) + list(W2.parameters()), lr=learning_rate)
criterion = nn.BCELoss()

# Training loop
losses = []
for epoch in range(epochs):
    # Forward pass
    hidden = activation(W1(X))
    output = torch.sigmoid(W2(hidden))

    loss = criterion(output, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Test predictions
hidden = activation(W1(X))
output = torch.sigmoid(W2(hidden))
print("Predictions:")
print(output.detach())

# Decision Boundary Plot
xx, yy = np.meshgrid(np.linspace(-1,2,200), np.linspace(-1,2,200))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

activation = nn.Tanh()
hidden = activation(W1(grid))
pred = torch.sigmoid(W2(hidden)).detach().numpy()
Z = pred.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.8)
plt.scatter(X[:,0], X[:,1], c=y[:,0], cmap='coolwarm', edgecolors='k')
plt.title("PyTorch XOR Decision Boundary")
plt.show()

# Training Loss Curve
plt.plot(losses)
plt.title("PyTorch Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
