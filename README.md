# Deep Learning Lab 4 - CNN from Scratch & YOLO Object Detection

**Student:** Aashish Joyson
**ID:** 2548533
**Course:** Deep Learning Laboratory

---

## Overview

This lab is divided into two parts:

1. **Part 1 — Manual CNN Implementation (NumPy, MNIST):** A Convolutional Neural Network built entirely from scratch using only NumPy. No deep learning frameworks. Every operation — convolution, ReLU, max pooling, flatten, fully connected layer, softmax, cross-entropy loss, and backpropagation — is implemented manually.

2. **Part 2 — YOLO Object Detection (YOLOv5 & YOLOv8):** Training and comparing two versions of the YOLO real-time object detector on the African Wildlife dataset, followed by an ensemble model that combines predictions from both.

---

## Part 1: Manual CNN from Scratch (NumPy + MNIST)

### Dataset

**MNIST** — Handwritten digit recognition

- 60,000 training images, 10,000 test images
- Images: 28×28 grayscale
- Classes: Digits 0–9 (10 classes)
- **Used in lab:** 1,000 training images, 200 test images (subset for manageable training time)

**Preprocessing:**
```python
X_train = X_train.astype(np.float32) / 255.0   # normalize pixel values to [0, 1]
```

---

### CNN Architecture (Manual, NumPy)

```
Input:         28×28 grayscale image
Conv Layer:    3 filters of size 3×3  → Output: (3, 26, 26)
ReLU:          Element-wise max(0, x) → Output: (3, 26, 26)
Max Pooling:   2×2 pool, stride 2     → Output: (3, 13, 13)
Flatten:       3 × 13 × 13 = 507 features
FC Layer:      507 → 10 (one score per digit class)
Softmax:       Convert scores to probabilities
Loss:          Cross-Entropy
```

---

### Implementation Details

#### 1. Convolution Forward (`conv_forward`)

```python
def conv_forward(image, filters):
    for n in range(num_filters):
        for i in range(out_h):
            for j in range(out_w):
                region = image[i:i+f, j:j+f]
                output[n, i, j] = np.sum(region * filters[n])
    return output
```

Slides each filter over the input image, computing the dot product between the filter and the local image patch. With a 3×3 filter on a 28×28 image: output size = (28−3+1) = 26×26 per filter.

#### 2. ReLU Activation

```python
def relu(feature_map):
    return np.maximum(0, feature_map)
```

Sets all negative values to zero. Introduces non-linearity after convolution. Without this, stacking layers would still be a linear transformation.

#### 3. Max Pooling (`max_pool`)

```python
def max_pool(feature_map, size=2):
    for n in range(num_filters):
        for i in range(out_h):
            for j in range(out_w):
                region = feature_map[n, i*size:(i+1)*size, j*size:(j+1)*size]
                pooled[n, i, j] = np.max(region)
    return pooled
```

Takes the maximum value from each 2×2 non-overlapping region. Reduces spatial size from 26×26 → 13×13 per filter. Makes the model translation-invariant — a shifted feature still produces the same pooled output.

#### 4. Flatten

```python
def flatten(feature_map):
    return feature_map.reshape(-1)   # (3, 13, 13) → (507,)
```

Converts the 3D feature map to a 1D vector to feed into the fully connected layer.

#### 5. Fully Connected Forward

```python
def fc_forward(x, W, b):
    return np.dot(x, W) + b   # (507,) · (507×10) + (10,) = (10,)
```

Standard linear layer: computes a score for each of the 10 digit classes.

#### 6. Softmax

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))   # subtract max for numerical stability
    return exp_x / np.sum(exp_x)
```

Converts raw scores (logits) into probabilities that sum to 1. The `- np.max(x)` trick prevents overflow from large exponentials.

#### 7. Cross-Entropy Loss

```python
def cross_entropy_loss(probs, label):
    return -np.log(probs[label] + 1e-9)
```

Measures how wrong the prediction is for the correct class. `1e-9` added to prevent `log(0)`. If the model assigns high probability to the correct class, loss is near 0. If it assigns near 0, loss explodes.

#### 8. Backpropagation (FC Layer Only)

```python
def fc_backward(x, probs, label, W):
    y_onehot = np.zeros_like(probs)
    y_onehot[label] = 1

    dL_dz = probs - y_onehot     # gradient of softmax + cross-entropy combined
    dW = np.outer(x, dL_dz)      # gradient w.r.t. weights
    db = dL_dz                    # gradient w.r.t. biases
    return dW, db
```

The combined gradient of cross-entropy loss through softmax is simply `probs − one_hot(label)`. This is the key identity that makes softmax + cross-entropy mathematically clean. Only the FC layer's weights and biases are updated — the conv filters are fixed.

---

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Training samples | 1,000 |
| Test samples | 200 |
| Filters | 3 × (3×3) |
| FC size | 507 → 10 |
| Learning rate | 0.01 |
| Epochs | 20 |
| Optimizer | Manual SGD |

---

### Training Results

| Epoch | Loss |
|-------|------|
| 1 | 1330.71 |
| 5 | 926.03 |
| 10 | 720.12 |
| 15 | 615.78 |
| 20 | 551.14 |

Loss decreased steadily from **1330 → 551** over 20 epochs, confirming the model was learning.

**Test Accuracy: 84.5%** (on 200 test images)

---

## Part 2: YOLO Object Detection — African Wildlife

### Dataset

**African Wildlife Dataset** (Ultralytics)

- **Classes:** Buffalo, Elephant, Rhino, Zebra (4 classes)
- **Format:** YOLOv5/v8 compatible (images + YAML config)
- **Task:** Real-time object detection with bounding boxes

---

### YOLOv8 Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # pretrained YOLOv8 Nano (transfer learning)

results = model.train(
    data="african-wildlife/african-wildlife.yaml",
    epochs=10,
    imgsz=640
)
```

**YOLOv8** is the latest YOLO generation — a unified, anchor-free detection framework. Starting from `yolov8n.pt` (pretrained on COCO) and fine-tuning on the wildlife dataset. This is transfer learning.

---

### YOLOv5 Training

```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 10 \
  --data african-wildlife/african-wildlife.yaml \
  --weights yolov5s.pt
```

**YOLOv5** uses anchor-based detection. Starting from `yolov5s.pt` (Small pretrained model). Cloned directly from `ultralytics/yolov5` GitHub repository.

---

### Inference

```bash
# YOLOv5
python detect.py --weights runs/train/exp2/weights/best.pt \
                 --source african-wildlife/images/test \
                 --conf 0.25

# YOLOv8 (Python API)
results = model(image_path)[0]
for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    cls, conf = int(box.cls[0]), float(box.conf[0])
```

Confidence threshold of 0.25 — detections below 25% confidence are discarded.

---

### Ensemble Model

```python
def detect_ensemble(image_path):
    # Draw YOLOv8 boxes in blue
    results8 = yolo8(image_path)[0]
    for box in results8.boxes:
        cv2.rectangle(img, ..., (255, 0, 0), 2)   # blue

    # Draw YOLOv5 boxes in green
    results5 = yolo5(image_path)
    for *xyxy, conf, cls in results5.xyxy[0]:
        cv2.rectangle(img, ..., (0, 255, 0), 2)   # green
```

Both models are run on the same image and their detections are overlaid. When both agree on a detection, confidence increases. Where they differ, the ensemble covers more ground — reducing missed detections from either individual model.

---

### YOLOv5 vs YOLOv8 Comparison

| Feature | YOLOv5 | YOLOv8 |
|---------|--------|--------|
| Detection approach | Anchor-based | Anchor-free |
| Bounding boxes | Tight, precise | Smooth, slightly larger |
| Convergence | Steady but slower | Faster, smoother |
| Scale handling | Good | Better (small/distant objects) |
| Inference speed | Fast | Slightly faster |
| API | CLI + Python | Python-first unified API |

**Key finding:** YOLOv8 showed better robustness across varied lighting and scale. YOLOv5 provided more precise localization. The ensemble combined both strengths.

---

### Results Summary

**Manual CNN (Part 1):**
- Test Accuracy: **84.5%** on MNIST subset
- Loss reduced from 1330 → 551 over 20 epochs
- Only FC layer backpropagated (conv filters fixed)

**YOLO (Part 2):**
- Both YOLOv5 and YOLOv8 successfully detected all 4 wildlife classes
- YOLOv8 showed higher stability and faster convergence
- Ensemble model improved reliability over either model alone

---

## Files

| File | Description |
|------|-------------|
| `2548533_LAB4_DL.ipynb` | Jupyter notebook with full implementation and outputs |
| `2548533_lab4_dl.py` | Python script version of the same implementation |

---

## Requirements

```bash
pip install numpy matplotlib tensorflow ultralytics torch torchvision opencv-python
```

**Dependencies:**
- Python 3.7+
- NumPy (Part 1 — CNN from scratch)
- TensorFlow/Keras (MNIST dataset loader only)
- Ultralytics (`pip install ultralytics`) — YOLOv8 and African Wildlife dataset
- PyTorch + torchvision — YOLOv5 via torch.hub
- OpenCV — ensemble visualization
- matplotlib — plots

---

## Usage

### Part 1: Manual CNN
```bash
python 2548533_lab4_dl.py
```

### Part 2: YOLO (run inside notebook)
```bash
jupyter notebook 2548533_LAB4_DL.ipynb
```

---

## Author

**Aashish Joyson** (2548533)
Deep Learning Laboratory
SPR Course

---

## License

This project is for educational purposes as part of the Deep Learning Laboratory coursework.
