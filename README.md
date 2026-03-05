# Deep Learning Lab 5 - RNN vs LSTM for Character-Level Text Generation

**Student:** Aashish Joyson
**ID:** 2548533
**Course:** Deep Learning Laboratory

---

## Overview

This lab implements and compares two recurrent neural network architectures — **Vanilla RNN** and **LSTM** — for character-level text generation. Both models are trained on the same text corpus and evaluated on their ability to generate coherent text, with comparisons on training loss, perplexity, embedding quality, and hidden state behavior.

---

## Dataset

- **Source:** Custom text file (`input.txt`) — a text corpus on football
- **Preprocessing:** Text converted to lowercase, then mapped to integer indices via a character vocabulary
- **Sequence length:** 50 characters (input) → 1 character (target)
- **Step size:** 1 (sliding window with stride 1 across the entire text)

```python
chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}
```

Each training sample: 50 consecutive characters as input, the next character as the label.

---

## Model Architectures

### RNN Model

```
Embedding:    vocab_size → 128 dimensions
RNN:          128 hidden units, 2 layers, batch_first=True
FC Output:    128 → vocab_size (one score per character)
```

```python
class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
```

The vanilla RNN processes the sequence step by step, maintaining a hidden state that gets updated at each time step. The final hidden state is passed through a fully connected layer to predict the next character.

### LSTM Model

```
Embedding:    vocab_size → 128 dimensions
LSTM:         128 hidden units, 2 layers, batch_first=True
FC Output:    128 → vocab_size (one score per character)
```

```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
```

The LSTM uses **gates** (forget, input, output) and a **cell state** to selectively remember or forget information across long sequences. This solves the vanishing gradient problem that limits vanilla RNNs.

---

## Training

```python
def train_model(model, X, y, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 0.002) |
| Loss | CrossEntropyLoss |
| Epochs | 100 |
| Hidden size | 128 |
| Num layers | 2 |
| Sequence length | 50 |

---

## Text Generation

```python
def generate_text(model, start_text, length=200):
    for _ in range(length):
        output = model(input_seq)
        prob = torch.softmax(output, dim=1).detach().cpu().numpy()
        next_char = np.random.choice(vocab_size, p=prob[0])
        generated += idx2char[next_char]
```

- Starts with a seed phrase (e.g., "In football we need to understand ")
- At each step: model predicts probability distribution over all characters
- Sampling from the distribution (not argmax) — introduces diversity in generated text
- Generated 1000 characters from each model for comparison

---

## Evaluation & Visualization

### 1. Loss vs Epoch (RNN vs LSTM)

Both models' training losses plotted across 100 epochs. LSTM typically converges faster and achieves lower final loss due to its ability to capture long-range dependencies.

### 2. Perplexity vs Epoch

```python
perplexity = np.exp(loss)
```

Perplexity measures how "confused" the model is — lower is better. A perplexity of 10 means the model is as uncertain as if it were choosing uniformly among 10 characters. LSTM achieves lower perplexity, indicating better language modeling.

### 3. Embedding Visualization (PCA)

```python
embeddings = model.embedding.weight.detach().cpu().numpy()
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)
```

The learned character embeddings are projected to 2D using PCA. Characters with similar roles (vowels, consonants, punctuation) tend to cluster together, showing the model has learned meaningful character relationships.

### 4. Hidden State Activation Visualization

```python
def get_hidden_states(model, input_sample):
    x = model.embedding(input_sample)
    if isinstance(model, LSTMModel):
        out, (hn, cn) = model.lstm(x, (h0, c0))
    else:
        out, hn = model.rnn(x, h0)
    return out.squeeze(0).cpu().numpy()
```

Mean hidden state activations plotted across the 50 time steps. Shows how each model's internal state evolves as it reads the input sequence. LSTM maintains more stable activations over time due to its gating mechanism, while vanilla RNN activations can fluctuate more.

---

## RNN vs LSTM Comparison

| Aspect | Vanilla RNN | LSTM |
|--------|------------|------|
| Architecture | Single hidden state | Hidden state + cell state + 3 gates |
| Long-range dependencies | Poor (vanishing gradient) | Strong (gating mechanism) |
| Training convergence | Slower | Faster |
| Final loss | Higher | Lower |
| Perplexity | Higher | Lower |
| Generated text quality | Less coherent | More coherent |
| Hidden state stability | Fluctuating | More stable |
| Computational cost | Lower | Higher (3× more parameters per cell) |

---

## Key Concepts

- **Character-level language model:** Predicts the next character given a sequence of previous characters
- **Embedding layer:** Converts character indices to dense vectors that capture character similarity
- **Vanishing gradient:** In vanilla RNNs, gradients shrink exponentially through time steps, preventing learning of long-range patterns
- **LSTM gates:** Forget gate (what to discard), input gate (what to store), output gate (what to expose) — control information flow through time
- **Perplexity:** exp(loss) — measures how well the model predicts the next character; lower = better
- **Sampling vs argmax:** Sampling from the probability distribution produces diverse text; argmax produces repetitive but "safe" text

---

## File

| File | Description |
|------|-------------|
| `2548533_DL_LAB5_Aashish_Joyson.ipynb` | Jupyter notebook with full implementation and outputs |

---

## Requirements

```bash
pip install torch numpy matplotlib scikit-learn
```

**Dependencies:**
- Python 3.7+
- PyTorch
- NumPy
- matplotlib
- scikit-learn (for PCA visualization)

---

## Usage

```bash
jupyter notebook 2548533_DL_LAB5_Aashish_Joyson.ipynb
```

The notebook will:
1. Load and preprocess the text corpus
2. Train an RNN model for 100 epochs
3. Train an LSTM model for 100 epochs
4. Generate text using both models
5. Plot loss and perplexity comparisons
6. Visualize character embeddings via PCA
7. Visualize hidden state activations

---

## Author

**Aashish Joyson** (2548533)
Deep Learning Laboratory
SPR Course

---

## License

This project is for educational purposes as part of the Deep Learning Laboratory coursework.
