"""
KeywordIQ — Inference Engine
==============================
Handles loading the trained model artifacts and generating SEO keyword
predictions for new Amazon product titles.  The Streamlit app imports
all public functions from this module.  No training logic lives here;
this file is strictly inference-only.

Course: MAI417-3 Deep Learning | MSAIM | Christ University, Bangalore
"""

import os
import re
import pickle
import numpy as np


# ═════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ═════════════════════════════════════════════════════════════════════

def load_artifacts() -> tuple:
    """
    Load the trained Keras model, fitted tokenizer, keyword vocabulary,
    and longtail lookup dictionary from disk.

    Returns:
        Tuple of (model, tokenizer, idx_to_keyword, vocab_size, longtail_lookup).

    Raises:
        FileNotFoundError: If the .h5 model file is missing, with
            step-by-step instructions for completing training first.
    """
    model_path    = os.path.join('model', 'keywordiq_model.h5')
    tokenizer_path = os.path.join('data', 'processed', 'tokenizer.pkl')
    vocab_path    = os.path.join('data', 'processed', 'keyword_vocab.pkl')
    longtail_path = os.path.join('data', 'processed', 'longtail_lookup.pkl')

    # ── Friendly error if model hasn't been trained yet ──────────────
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "\n" + "=" * 60 + "\n"
            "MODEL NOT FOUND: model/keywordiq_model.h5\n"
            "=" * 60 + "\n"
            "Please follow these steps:\n"
            "1. Run: python training/preprocess.py\n"
            "2. Upload project folder to Google Drive\n"
            "3. Open training/colab_notebook.ipynb in Google Colab\n"
            "4. Run all cells (Runtime → Run All)\n"
            "5. Download keywordiq_model.h5 when prompted\n"
            "6. Place the downloaded file in the model/ folder\n"
            "7. Then restart: streamlit run app/streamlit_app.py\n"
            + "=" * 60
        )

    # ── Load Keras model ─────────────────────────────────────────────
    from tensorflow.keras.models import load_model
    model = load_model(model_path)

    # ── Load tokenizer ───────────────────────────────────────────────
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # ── Load keyword vocabulary ──────────────────────────────────────
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    idx_to_keyword = vocab_data['idx_to_keyword']
    vocab_size     = vocab_data['vocab_size']

    # ── Load longtail lookup (optional — may not exist) ──────────────
    longtail_lookup = {}
    if os.path.exists(longtail_path):
        with open(longtail_path, 'rb') as f:
            longtail_lookup = pickle.load(f)

    return model, tokenizer, idx_to_keyword, vocab_size, longtail_lookup


# ═════════════════════════════════════════════════════════════════════
# CLEAN TITLE  (must mirror training/preprocess.py exactly)
# ═════════════════════════════════════════════════════════════════════

def clean_title(text: str) -> str:
    """
    Apply the identical cleaning pipeline used during preprocessing so
    that inference input matches the distribution the model was trained on.
    Any mismatch between training-time and inference-time cleaning will
    degrade prediction quality.

    Steps:
        1. Lowercase
        2. Strip HTML tags
        3. Keep only a-z, 0-9, spaces, and hyphens
        4. Collapse whitespace
        5. Strip leading/trailing spaces

    Args:
        text: Raw product title string.

    Returns:
        Cleaned, lowercased string.
    """
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)            # remove HTML tags
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)    # keep safe chars only
    text = re.sub(r'\s+', ' ', text)               # collapse whitespace
    return text.strip()


# ═════════════════════════════════════════════════════════════════════
# SINGLE-TITLE PREDICTION
# ═════════════════════════════════════════════════════════════════════

def predict_keywords(title: str,
                     model,
                     tokenizer,
                     idx_to_keyword: dict,
                     threshold: float = 0.3) -> list:
    """
    Predict SEO keywords for a single Amazon product title.

    The function cleans the title, tokenizes it, runs it through the
    trained model, and returns all keywords whose sigmoid probability
    exceeds the given threshold.  Two fallback strategies ensure that
    the caller always gets at least some results.

    Args:
        title:          Raw product title string (cleaned internally).
        model:          Loaded Keras model.
        tokenizer:      Fitted Keras Tokenizer.
        idx_to_keyword: Dict mapping integer index → keyword string.
        threshold:      Minimum sigmoid probability to include a keyword
                        (default 0.3 — lower = more keywords).

    Returns:
        List of tuples [(keyword, confidence), ...] sorted by confidence
        descending.  Maximum 10 entries.
        Example: [("body wash", 0.92), ("moisturizer", 0.87)]
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    MAX_SEQ_LENGTH = 60  # must match preprocess.py / train.py

    # Clean → tokenize → pad
    cleaned  = clean_title(title)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(sequence, maxlen=MAX_SEQ_LENGTH,
                             padding='post', truncating='post')

    # Forward pass — output shape: (1, vocab_size)
    probabilities = model.predict(padded, verbose=0)[0]

    # ── Primary extraction: all keywords above threshold ─────────────
    results = []
    for idx, prob in enumerate(probabilities):
        if prob >= threshold:
            keyword = idx_to_keyword.get(idx, None)
            if keyword:
                results.append((keyword, float(prob)))

    results.sort(key=lambda x: x[1], reverse=True)

    # ── Fallback 1: halve threshold and retry ────────────────────────
    if len(results) == 0:
        fallback_threshold = threshold * 0.5
        for idx, prob in enumerate(probabilities):
            if prob >= fallback_threshold:
                keyword = idx_to_keyword.get(idx, None)
                if keyword:
                    results.append((keyword, float(prob)))
        results.sort(key=lambda x: x[1], reverse=True)

    # ── Fallback 2: force top-3 regardless of threshold ──────────────
    if len(results) == 0:
        top_indices = np.argsort(probabilities)[-3:][::-1]
        for idx in top_indices:
            keyword = idx_to_keyword.get(int(idx), None)
            if keyword:
                results.append((keyword, float(probabilities[idx])))

    return results[:10]


# ═════════════════════════════════════════════════════════════════════
# BATCH PREDICTION
# ═════════════════════════════════════════════════════════════════════

def predict_batch(titles: list,
                  model,
                  tokenizer,
                  idx_to_keyword: dict,
                  threshold: float = 0.3) -> list:
    """
    Predict keywords for many titles in one batch operation.  Significantly
    faster than calling predict_keywords() in a loop because the GPU
    processes all titles in a single forward pass per chunk.

    Args:
        titles:         List of raw product title strings.
        model:          Loaded Keras model.
        tokenizer:      Fitted Keras Tokenizer.
        idx_to_keyword: Dict mapping index → keyword string.
        threshold:      Minimum probability to include a keyword.

    Returns:
        List of strings — one comma-separated keyword string per title.
        Example: ["body wash, moisturizer", "nail glue, nail adhesive"]
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    MAX_SEQ_LENGTH = 60

    # Clean all titles
    cleaned_titles = [clean_title(t) for t in titles]

    # Tokenize all at once
    sequences = tokenizer.texts_to_sequences(cleaned_titles)
    padded    = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH,
                              padding='post', truncating='post')

    # Single batch forward pass — shape: (len(titles), vocab_size)
    all_probabilities = model.predict(padded, batch_size=64, verbose=0)

    results = []
    for prob_vector in all_probabilities:
        keywords = []
        for idx, prob in enumerate(prob_vector):
            if prob >= threshold:
                keyword = idx_to_keyword.get(idx, None)
                if keyword:
                    keywords.append((keyword, float(prob)))

        keywords.sort(key=lambda x: x[1], reverse=True)

        # Fallback: if nothing exceeds threshold, take the top prediction
        if len(keywords) == 0:
            top_idx = int(np.argmax(prob_vector))
            top_keyword = idx_to_keyword.get(top_idx, 'general product')
            keywords = [(top_keyword, float(prob_vector[top_idx]))]

        # Format as comma-separated string (max 5 keywords per title)
        keyword_string = ', '.join([kw for kw, _ in keywords[:5]])
        results.append(keyword_string)

    return results
