import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype="float32")
y = np.array([[0],[1],[1],[0]], dtype="float32")

# Hyperparameters
learning_rate = 0.5
hidden_neurons = 8
epochs = 10000

# Initialize weights and biases
W1 = tf.Variable(tf.random.normal([2, hidden_neurons]))
b1 = tf.Variable(tf.zeros([hidden_neurons]))

W2 = tf.Variable(tf.random.normal([hidden_neurons, 1]))
b2 = tf.Variable(tf.zeros([1]))

# Forward pass function
def forward(x):
    h = tf.tanh(tf.matmul(x, W1) + b1)
    o = tf.sigmoid(tf.matmul(h, W2) + b2)
    return o

# Training loop
losses = []
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        pred = forward(X)
        loss = tf.reduce_mean((pred - y)**2)

    grads = tape.gradient(loss, [W1, b1, W2, b2])
    for var, grad in zip([W1, b1, W2, b2], grads):
        var.assign_sub(learning_rate * grad)
    losses.append(loss.numpy())

# Predictions
print("Predictions:")
print(forward(X).numpy())

# Decision Boundary Plot
xx, yy = np.meshgrid(np.linspace(-1,2,200), np.linspace(-1,2,200))
grid = np.c_[xx.ravel(), yy.ravel()].astype("float32")

pred = forward(grid).numpy()
Z = pred.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.8)
plt.scatter(X[:,0], X[:,1], c=y[:,0], cmap='coolwarm', edgecolors='k')
plt.title("TensorFlow (Low-Level) XOR Decision Boundary")
plt.show()

# Training Loss Curve
plt.plot(losses)
plt.title("TensorFlow (Low-Level) Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
