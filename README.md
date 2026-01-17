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

---

## Lab 2: Fashion MNIST Classification with Neural Networks

**Student:** Aashish Joyson
**ID:** 2548533
**Course:** Deep Learning Laboratory

### Overview

This lab implements a neural network classifier for the Fashion MNIST dataset using PyTorch. The project explores various aspects of deep learning including:
- Neural network architecture design
- Training and evaluation workflows
- Activation function comparison
- Overfitting detection and analysis
- Network activation visualization

---

### Dataset

**Fashion MNIST** contains 70,000 grayscale images (28x28 pixels) of 10 different clothing categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

**Split:**
- Training set: 60,000 images
- Test set: 10,000 images

---

### Model Architecture

#### FashionClassifier (Funnel Design)
A 4-layer fully connected neural network with progressively decreasing layer sizes:

```
Input Layer:    784 neurons (28×28 flattened image)
Hidden Layer 1: 256 neurons + ReLU
Hidden Layer 2: 128 neurons + ReLU
Hidden Layer 3:  64 neurons + ReLU
Output Layer:    10 neurons (10 classes)
```

**Architecture Features:**
- Funnel design: Progressively compresses information from 784 → 10
- ReLU activation for non-linearity
- CrossEntropyLoss for multi-class classification
- Adam optimizer with learning rate 0.001

---

### Experiments and Results

#### Experiment 1: Basic Training (10 Epochs)

**Training Progress:**
```
Epoch [1/10]  -> Loss: 0.5229 | Accuracy: 80.86%
Epoch [2/10]  -> Loss: 0.3794 | Accuracy: 86.01%
Epoch [3/10]  -> Loss: 0.3381 | Accuracy: 87.63%
Epoch [4/10]  -> Loss: 0.3157 | Accuracy: 88.28%
Epoch [5/10]  -> Loss: 0.2949 | Accuracy: 89.10%
Epoch [6/10]  -> Loss: 0.2787 | Accuracy: 89.67%
Epoch [7/10]  -> Loss: 0.2641 | Accuracy: 90.16%
Epoch [8/10]  -> Loss: 0.2503 | Accuracy: 90.69%
Epoch [9/10]  -> Loss: 0.2415 | Accuracy: 90.86%
Epoch [10/10] -> Loss: 0.2290 | Accuracy: 91.41%
```

**Final Results:**
- **Training Accuracy:** 91.41%
- **Test Accuracy:** 87.73%
- **Generalization Gap:** 3.68%

**Conclusion:** The model generalizes well with no significant overfitting detected.

---

#### Experiment 2: Activation Function Comparison (20 Epochs)

Compared three activation functions: **ReLU**, **Sigmoid**, and **Tanh**

**Key Findings:**

| Activation | Convergence Speed | Final Performance | Issues |
|------------|------------------|-------------------|--------|
| **ReLU** | ✅ Fast | ✅ Best | None |
| **Tanh** | 🟡 Moderate | 🟡 Good | Slower than ReLU |
| **Sigmoid** | ❌ Very Slow | ❌ Poor | Vanishing gradient problem |

**Winner: ReLU**
- Fastest convergence
- No vanishing gradient issues
- Best choice for deep networks

**Sigmoid Struggled:** The sigmoid function showed very slow learning in early epochs due to the vanishing gradient problem, making it unsuitable for deep networks.

---

#### Experiment 3: Overfitting Stress Test (20 Epochs)

Extended training to 20 epochs to observe overfitting behavior.

**Key Observations:**

**Epoch 1-10:** Both training and test loss decrease together → Healthy learning

**Epoch 12+:**
- Training loss continues to decrease
- **Test loss starts increasing** ⚠️ (Overfitting detected!)

**Final Metrics:**
- **Training Accuracy:** Continues improving
- **Test Accuracy:** Plateaus and slightly degrades
- **Overfitting Point:** Around Epoch 12

**Conclusion:** The optimal stopping point is **10 epochs**. Beyond this, the model begins memorizing the training data rather than learning generalizable patterns.

---

### Advanced Task: Network Activation Visualization

Visualized internal layer activations to understand what the network "sees":

**Layer 1 (256 neurons):**
- Detects basic features: edges, lines, contours
- Recognizes fundamental shape outlines (e.g., shoe silhouette)

**Layer 2 (128 neurons):**
- Combines basic features into patterns
- Recognizes textures and partial shapes

**Layer 3 (64 neurons):**
- High-level abstract features
- Class-discriminative patterns
- Final decision-making representations

**Insight:** The network follows a hierarchical feature learning approach—from simple edges to complex semantic patterns.

---

### Requirements

```bash
pip install torch torchvision matplotlib numpy
```

**Dependencies:**
- Python 3.7+
- PyTorch 1.9+
- torchvision
- matplotlib
- numpy

---

### Usage

#### Run the complete experiment:
```bash
python aashishjoyson_2548533_dl_lab2.py
```

The script will:
1. Download Fashion MNIST dataset automatically
2. Train the base model (10 epochs)
3. Generate training loss and accuracy plots
4. Evaluate on test set
5. Run activation function comparison
6. Perform overfitting stress test
7. Visualize network activations

---

### Key Takeaways

1. **Architecture Matters:** The funnel design (784→256→128→64→10) effectively compresses spatial information for classification

2. **ReLU is Superior:** For deep networks, ReLU outperforms sigmoid and tanh due to its resistance to vanishing gradients

3. **Early Stopping is Critical:** Training beyond 10 epochs causes overfitting. Monitoring validation loss is essential

4. **Hierarchical Learning:** Neural networks learn in stages—low-level features (edges) → mid-level patterns → high-level semantic concepts

5. **Generalization Check:** A 3-4% gap between train and test accuracy indicates healthy generalization

---

### Results Summary

| Metric | Value |
|--------|-------|
| Final Training Accuracy | 91.41% |
| Final Test Accuracy | 87.73% |
| Optimal Epochs | 10 |
| Best Activation | ReLU |
| Model Parameters | ~235,000 |
| Training Time | ~2-3 minutes (CPU) |

---

### File Structure

```
SPR-2548533/
├── LAB1/                                  # Lab 1 branch
│   ├── keras.py
│   ├── pytorch.py
│   └── tensorflow.py
└── LAB2/                                  # Lab 2 branch (this)
    ├── aashishjoyson_2548533_dl_lab2.py
    └── README.md
```

---

### Future Improvements

1. Add dropout layers to reduce overfitting
2. Implement learning rate scheduling
3. Try convolutional layers (CNN) for better feature extraction
4. Experiment with batch normalization
5. Add data augmentation (rotation, shift, zoom)

---

### Author

**Aashish Joyson** (2548533)
Deep Learning Laboratory
SPR Course

---

### License

This project is for educational purposes as part of the Deep Learning Laboratory coursework.
