import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype="float32")
y = np.array([[0],[1],[1],[0]], dtype="float32")

# Build model
model = keras.Sequential([
    layers.Dense(8, activation="tanh", input_shape=(2,)),
    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.5),
    loss="binary_crossentropy"
)

# Train model
history = model.fit(X, y, epochs=10000, verbose=0)

# Predictions
print("Predictions:")
print(model.predict(X))

# Decision Boundary Plot
xx, yy = np.meshgrid(np.linspace(-1,2,200), np.linspace(-1,2,200))
grid = np.c_[xx.ravel(), yy.ravel()]
pred = model.predict(grid)
Z = pred.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.8)
plt.scatter(X[:,0], X[:,1], c=y[:,0], cmap='coolwarm', edgecolors='k')
plt.title("Keras XOR Decision Boundary")
plt.show()

# Training Loss Curve
plt.plot(history.history['loss'])
plt.title("Keras Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
