# Deep Learning Lab - SPR-2548533

## Lab 1: XOR Neural Network Implementation

This lab implements a simple neural network to solve the XOR problem using three different deep learning frameworks.

### Implementations

1. **Keras** ([keras.py](keras.py))
   - Uses Keras Sequential API
   - 8 hidden neurons with tanh activation
   - Sigmoid output layer
   - Adam optimizer with learning rate 0.5

2. **PyTorch** ([pytorch.py](pytorch.py))
   - Manual implementation using PyTorch layers
   - 8 hidden neurons with tanh activation
   - Sigmoid output layer
   - SGD optimizer with learning rate 0.5

3. **TensorFlow** ([tensorflow.py](tensorflow.py))
   - Low-level TensorFlow implementation
   - Manual weight initialization and gradient descent
   - 8 hidden neurons with tanh activation
   - Sigmoid output layer

### Requirements

```bash
pip install numpy matplotlib tensorflow torch
```

### Running the Code

Each file can be run independently:

```bash
python keras.py
python pytorch.py
python tensorflow.py
```

### Output

Each implementation produces:
- Predictions for the XOR problem
- Decision boundary visualization
- Training loss curve

### XOR Problem

The XOR (exclusive OR) problem is a classic non-linearly separable problem:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |
