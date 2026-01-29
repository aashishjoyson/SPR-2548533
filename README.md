# Deep Learning Lab 3 - Regularization Techniques

**Student:** Aashish Joyson
**ID:** 2548533
**Course:** Deep Learning Laboratory

---

## Overview

This lab implements and compares regularization techniques applied to a neural network binary classifier. The project explores how different regularization strategies affect training dynamics, weight distribution, and generalization performance.

---

## Dataset

**Breast Cancer Wisconsin Dataset** (from scikit-learn)

- **Total samples:** 569
- **Features:** 30 numerical features (mean, standard error, worst values of cell nucleus properties)
- **Task:** Binary classification — Malignant (0) vs Benign (1)

**Split:**
- Training set: 455 samples
- Test set: 114 samples

---

## Model Architecture

### NeuralNet (Binary Classifier)

```
Input Layer:    30 neurons (breast cancer features)
Hidden Layer 1: 64 neurons + ReLU
Hidden Layer 2: 32 neurons + ReLU
Output Layer:    1 neuron  + Sigmoid (binary classification)
```

- **Loss Function:** Binary Cross Entropy (BCELoss)
- **Optimizer:** Adam (lr = 0.001)
- **Training Epochs:** 60

---

## Experiments

### Experiment 1: No Regularization

Baseline model trained without any regularization penalty.

```
Epoch [10/60] | Train Loss: 0.0561 | Test Loss: 0.0556 | Test Acc: 0.9825
Epoch [30/60] | Train Loss: 0.0180 | Test Loss: 0.0693 | Test Acc: 0.9737
Epoch [60/60] | Train Loss: 0.0044 | Test Loss: 0.0879 | Test Acc: 0.9825
```

**Observation:** Training loss continuously drops while test loss rises — classic overfitting.

---

### Experiment 2: L2 Regularization (weight_decay = 0.01)

L2 penalty discourages large weights via the Adam optimizer's `weight_decay` parameter.

```
Epoch [10/60] | Train Loss: 0.0718 | Test Loss: 0.0686 | Test Acc: 0.9912
Epoch [30/60] | Train Loss: 0.0465 | Test Loss: 0.0577 | Test Acc: 0.9912
Epoch [60/60] | Train Loss: 0.0397 | Test Loss: 0.0587 | Test Acc: 0.9912
```

**Observation:** Stable test loss throughout training. Best generalization.

---

### Experiment 3: L1 Regularization (l1_lambda = 0.0005)

L1 penalty added manually to loss, encouraging sparse weight distributions.

```
Epoch [10/60] | Train Loss: 0.0795 | Test Loss: 0.0631 | Test Acc: 0.9825
Epoch [30/60] | Train Loss: 0.0362 | Test Loss: 0.0528 | Test Acc: 0.9825
Epoch [60/60] | Train Loss: 0.0257 | Test Loss: 0.0563 | Test Acc: 0.9737
```

**Observation:** High sparsity (2427/4000 weights near zero). Slightly lower accuracy due to aggressive feature selection.

---

### Experiment 4: Elastic Net (l1_lambda = 0.0005, l2_lambda = 0.01)

Combines both L1 and L2 penalties for balanced regularization.

```
Epoch [10/60] | Train Loss: 0.0934 | Test Loss: 0.0813 | Test Acc: 0.9912
Epoch [30/60] | Train Loss: 0.0619 | Test Loss: 0.0680 | Test Acc: 0.9912
Epoch [60/60] | Train Loss: 0.0507 | Test Loss: 0.0606 | Test Acc: 0.9912
```

**Observation:** Highest sparsity (3040/4000 weights near zero) with strong test accuracy.

---

## Results Summary

| Method        | Train Accuracy | Test Accuracy | Test Loss | Near-Zero Weights |
|---------------|---------------|--------------|-----------|------------------|
| No Reg        | 99.78%        | 98.25%       | 0.0879    | 18 / 4000        |
| L2            | 99.12%        | **99.12%**   | 0.0587    | 168 / 4000       |
| L1            | 99.34%        | 97.37%       | 0.0563    | 2427 / 4000      |
| Elastic Net   | 98.90%        | **99.12%**   | 0.0606    | 3040 / 4000      |

---

## Weight Statistics Comparison

| Method      | Mean     | Std Dev  | Near-Zero |
|-------------|----------|----------|-----------|
| No Reg      | 0.0174   | 0.1386   | 18        |
| L2          | 0.0090   | 0.0522   | 168       |
| L1          | 0.0101   | 0.0710   | 2427      |
| Elastic Net | 0.0052   | 0.0480   | 3040      |

---

## Key Takeaways

1. **No Regularization → Overfitting:** Training loss drops aggressively but test loss rises, indicating memorization
2. **L2 → Best Generalization:** Stable test accuracy and loss throughout training; controls weight magnitude
3. **L1 → Sparsity:** Pushes weights to zero for implicit feature selection, slightly lower accuracy
4. **Elastic Net → Best of Both:** Combines sparsity from L1 and weight control from L2, achieves highest sparsity with strong accuracy

---

## Requirements

```bash
pip install torch torchvision matplotlib scikit-learn numpy
```

**Dependencies:**
- Python 3.7+
- PyTorch
- scikit-learn
- matplotlib
- numpy

---

## Usage

Open and run the notebook:

```bash
jupyter notebook 2548533_DL_Lab3.ipynb
```

The notebook will:
1. Load and preprocess the Breast Cancer dataset
2. Train models with No Reg, L2, L1, and Elastic Net regularization
3. Plot test loss and accuracy comparisons
4. Analyze weight distributions and sparsity
5. Print final performance summary

---

## Author

**Aashish Joyson** (2548533)
Deep Learning Laboratory
SPR Course

---

## License

This project is for educational purposes as part of the Deep Learning Laboratory coursework.
