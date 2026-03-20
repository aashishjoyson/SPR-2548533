# 🔍 KeywordIQ — AI-Powered SEO Keyword Generator

> **Automatically generate high-quality SEO keywords from Amazon Health & Supplement product titles using a trained deep learning model.**

Built for the **NeuralHack 2026** End-Trimester Hackathon  
Course: **MAI417-3 Deep Learning** | Programme: **M.Sc. AIML** | **Christ (Deemed to be University), Bangalore**

---

## 📌 What Is KeywordIQ?

KeywordIQ is an end-to-end deep learning application that solves a real problem faced by **Amazon resellers and dropshippers** in the Health & Supplements category.

When a reseller sources health products — protein powders, vitamins, probiotics, sleep aids, collagen supplements — they need relevant SEO keywords for each product listing to appear in search results. Writing keywords manually for hundreds of products is slow and doesn't scale.

**KeywordIQ solves this:** paste any health product title, get instant SEO keywords with confidence scores. Upload a CSV of 500 titles, download it with a keywords column added — in seconds.

---

## 🎯 Problem Statement

**Input:** A raw Amazon health product title  
```
Optimum Nutrition Gold Standard Whey Protein Powder Double Chocolate 5 Pound
```

**Output:** Ranked SEO keywords with confidence scores  
```
protein powder (96%)  |  whey protein (95%)  |  muscle protein (74%)
```

**Task Type:** Multi-Label Binary Classification — multiple keywords can apply to one product simultaneously. Each keyword is an independent yes/no prediction, not a competition between classes.

---

## 🧠 Model Architecture

KeywordIQ uses a **Parallel CNN-BiLSTM architecture** with an MLP classification head.

```
Product Title (raw text)
        ↓
  [Preprocessing] → lowercase → clean → tokenize → pad to 60 tokens
        ↓
  [Shared Embedding Layer] → 10,000 vocab × 64 dimensions
        ↓                           ↓
  [CNN Branch]               [BiLSTM Branch]
  Conv1D(128, k=3)           Bidirectional(LSTM(64))
  Conv1D(64,  k=2)           Forward + Backward reads
  GlobalMaxPool1D            128-dim context vector
  64-dim vector                      ↓
        └──────── Concatenate ───────┘
                      ↓
              192-dim joint vector
                      ↓
           [Dense(256) + Dropout(0.4)]
           [Dense(128) + Dropout(0.3)]
                      ↓
        [Dense(500, sigmoid output)]
                      ↓
     500 independent keyword probabilities
                      ↓
   threshold = 0.30 → final keyword list
```

### Why This Architecture?

| Component | What It Captures | Why Needed |
|---|---|---|
| **Conv1D (k=3)** | 3-word local phrases: "whey protein powder", "vitamin d3 supplement" | Product titles have predictable short keyword phrases |
| **Conv1D (k=2)** | 2-word patterns: "protein powder", "immune support", "pain relief" | Bigrams are the most common SEO keyword format |
| **Bidirectional LSTM** | Full title meaning: "this is a supplement for sleep support" | Overall product category requires sequential context |
| **Shared Embedding** | Consistent word representations for both branches | Prevents contradictory word meanings across branches |
| **Dropout (0.4, 0.3)** | Regularization — prevents memorization | Validated: Val AUC > Train AUC throughout training |
| **Sigmoid (not Softmax)** | Independent probability per keyword | Multiple keywords apply simultaneously — not mutually exclusive |

---

## 📊 Training Results

| Metric | Value |
|---|---|
| **Best Validation AUC** | **0.9668** |
| **Best Validation Accuracy** | **99.67%** |
| **Best Epoch** | 22 / 30 |
| **Early Stopping** | Triggered at Epoch 27 |
| **Overfitting** | None — Val AUC ≥ Train AUC throughout |
| **Total Parameters** | 894,068 (3.41 MB) |

### Training Progression

| Epoch | Train AUC | Val AUC | Status |
|---|---|---|---|
| 1 | 0.498 | 0.500 | Initial — learning starts |
| 8 | 0.561 | 0.624 | Pattern recognition begins |
| 10 | 0.801 | 0.858 | Strong generalization |
| 11 | 0.869 | 0.923 | Excellent performance |
| 15 | 0.944 | 0.955 | Near-optimal |
| **22** | **0.963** | **0.9668** | **Best model saved** |
| 27 | — | — | EarlyStopping triggered |

---

## 🗂️ Project Structure

```
keywordiq/
├── data/
│   ├── raw/
│   │   └── products_merged.csv        ← your product data goes here
│   └── processed/
│       ├── X_train.npy                ← (34296, 60) token sequences
│       ├── X_val.npy                  ← (6053, 60) token sequences
│       ├── y_train.npy                ← (34296, 500) multi-hot labels
│       ├── y_val.npy                  ← (6053, 500) multi-hot labels
│       ├── tokenizer.pkl              ← fitted Keras Tokenizer
│       ├── keyword_vocab.pkl          ← {keyword ↔ index} mappings
│       └── longtail_lookup.pkl        ← title → phrase lookup
├── model/
│   └── keywordiq_model.h5             ← trained model weights
├── training/
│   ├── preprocess.py                  ← 10-step data pipeline
│   ├── train.py                       ← model definition + training
│   └── colab_notebook.ipynb           ← ready-to-run Colab notebook
├── app/
│   └── streamlit_app.py               ← full Streamlit UI (3 tabs)
├── utils/
│   └── inference.py                   ← prediction engine
└── requirements.txt
```

---

## 📦 Dataset

- **Domain:** Amazon Health & Supplements category
- **Total rows collected:** 142,238 product title → keyword mapping pairs
- **Files merged:** 5 source files (CSV and XLSX formats)
- **After preprocessing:** 40,349 clean unique samples
- **Training / Validation split:** 85% / 15%  →  34,296 training | 6,053 validation
- **Keyword vocabulary:** Top-500 most frequent SEO keywords
- **Sequence length:** 60 tokens (covers 95%+ of product titles)
- **Tokenizer vocabulary:** 10,000 most frequent words + OOV token

### Sample Data

| Product Title | Keywords |
|---|---|
| Optimum Nutrition Whey Protein Powder Double Chocolate 5lb | protein powder, whey protein, muscle protein |
| Nature Made Vitamin D3 2000 IU Softgels Immune Support 260ct | vitamin d3, immune support, dietary supplements |
| Natrol Melatonin Sleep Aid Gummies Strawberry 10mg 90ct | sleep aid, melatonin gummies, sleep supplements |
| Vital Proteins Collagen Peptides Powder Vanilla 24 Oz | collagen supplements, collagen powder, protein powder |
| Align Probiotic Daily Digestive Health 24 Hour Support 56ct | probiotic supplements, digestive health, gut health |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/keywordiq.git
cd keywordiq
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data

Place your product data CSV in `data/raw/products_merged.csv`. The CSV must have these columns:

```
Product Title          | Keyword_Broad_Options              | keyword_longtail
-----------------------|------------------------------------|---------------------------
Whey Protein 5lb       | ['protein powder', 'whey protein'] | protein powder for muscle
Vitamin D3 2000 IU     | ['vitamin d3', 'immune support']   | vitamin d for immune health
```

### 4. Run Preprocessing (Local)

```bash
python training/preprocess.py
```

This runs the full 10-step pipeline and saves processed files to `data/processed/`.

### 5. Train the Model (Google Colab Recommended)

Upload the `keywordiq/` folder to Google Drive, then:

1. Open `training/colab_notebook.ipynb` in [Google Colab](https://colab.research.google.com)
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Click **Runtime → Run All**
4. Training takes approximately 20–30 minutes
5. Download `keywordiq_model.h5` when prompted
6. Place it in `model/`

### 6. Launch the App

```bash
python -m streamlit run app/streamlit_app.py
```

App opens at `http://localhost:8501`

---

## 🖥️ App Features

### Tab 1 — Single Title Prediction

- Paste any health product title
- Get instant keyword predictions with confidence percentages
- Color-coded keyword badges (green = high confidence, blue = medium)
- Confidence bar chart with adjustable threshold slider
- Copy-ready keyword lists in two formats

### Tab 2 — Batch CSV Processing

- Upload a CSV file with a `title` column
- App processes all titles simultaneously using vectorized batch inference
- Real-time progress bar
- Download enriched CSV with `predicted_keywords` column added
- Handles hundreds of products in seconds

### Tab 3 — Model Insights

- Training vs Validation AUC and Loss curves
- Key performance metrics at a glance
- Complete model architecture table
- Layer-by-layer parameter breakdown

---

## 🧪 Try These Titles

The model performs best on Health & Supplement product titles. Try these in the app:

```
Optimum Nutrition Gold Standard Whey Protein Powder Vanilla Ice Cream 2 Pound

Garden of Life Organic Vegan Pre Workout Energy Powder With Caffeine

Nature Made Vitamin D3 2000 IU Softgels For Immune Support 260 Count

Natrol Melatonin Sleep Aid Gummy Vitamins Strawberry 10mg 90 Count

Natural Vitality Calm Magnesium Stress Relief Supplement Powder Raspberry Lemon

Align Probiotic Supplement Daily Digestive Health 24 Hour Support 56 Capsules

Vital Proteins Collagen Peptides Powder With Hyaluronic Acid Vanilla 24 Oz

Neuriva Brain Performance Supplement With Vitamins B6 B12 60 Capsules

Emergen-C 1000mg Vitamin C Powder With Antioxidants B Vitamins Orange 60 Count

MuscleTech Nitro Tech Whey Protein Isolate Vanilla 4 Pound
```

---

## 🔬 Technical Details

### Loss Function

Binary Cross-Entropy — computed independently for each of the 500 output neurons:

```
L = -1/N × Σᵢ Σₖ [ yᵢₖ · log(ŷᵢₖ) + (1 - yᵢₖ) · log(1 - ŷᵢₖ) ]
```

### Why Not Softmax?

Softmax forces all output probabilities to sum to 1.0, implying that predicting one keyword reduces confidence in others. Since multiple keywords apply to one product simultaneously, each output must be independent. **Sigmoid on each neuron independently** is the mathematically correct formulation.

### Regularization

| Technique | Where Applied | Effect |
|---|---|---|
| Dropout (0.4) | After Dense(256) | Drops 40% of neurons during training |
| Dropout (0.3) | After Dense(128) | Drops 30% of neurons during training |
| L2 (1e-4) | Dense Layer weights | Penalizes large weight values |

Validation AUC consistently matched or exceeded Training AUC — confirming regularization worked correctly.

### Optimization

```python
optimizer = Adam(learning_rate=0.001)
callbacks = [
    EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True),
    ModelCheckpoint(save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
]
```

---

## 📋 Requirements

```
tensorflow==2.13.0
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
nltk==3.8.1
matplotlib==3.7.2
seaborn==0.12.2
openpyxl==3.1.2
```

---

## 🏫 Course Outcomes Covered

| CO | Description | Implementation |
|---|---|---|
| CO1 | Deep feedforward and backpropagation for classification | MLP head with 256→128→500 dense layers |
| CO2 | Regularization techniques for optimization | Dropout (0.4, 0.3) + L2 weight decay |
| CO3 | CNN and RNN for vision and sequence tasks | Conv1D (text n-grams) + Bidirectional LSTM |
| CO4 | Autoencoders for feature extraction | Shared embedding as compressed representation |
| CO5 | Compare architectures for performance | CNN-only vs LSTM-only vs parallel CNN-BiLSTM |

---

## 📁 Model Card

| Property | Value |
|---|---|
| Model Type | Parallel CNN-BiLSTM + MLP |
| Task | Multi-label text classification |
| Input | Product title string (max 60 tokens) |
| Output | 500-dimensional sigmoid probability vector |
| Training Data | 34,296 Amazon Health & Supplement listings |
| Val AUC | 0.9668 |
| Val Accuracy | 99.67% |
| Parameters | 894,068 (3.41 MB) |
| Framework | TensorFlow 2.13 / Keras Functional API |
| Training Platform | Google Colab T4 GPU |

---

## 👤 Author
**Mathi Aashish Joyson**
---
**M.Sc. AIML — PG III Trimester**  
Christ (Deemed to be University), Bangalore — 560 029  
Course: MAI417-3 Deep Learning  
Examination: NeuralHack 2026 — End-Trimester Hackathon, March 2026

---

*KeywordIQ — Turn Amazon titles into SEO gold.*
