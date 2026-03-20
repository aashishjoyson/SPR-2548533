"""
KeywordIQ — Preprocessing Script
=================================
This script loads the merged Amazon product dataset, cleans text data,
builds a keyword vocabulary, encodes multi-hot labels, tokenizes product
titles, and saves all processed artifacts to data/processed/.

Run this ONCE on your local machine before uploading to Google Colab
for model training.

Course: MAI417-3 Deep Learning | MSAIM | Christ University, Bangalore
"""

import os
import re
import ast
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS — must match training/train.py
# ─────────────────────────────────────────────────────────────────────
MAX_VOCAB_SIZE = 10000   # maximum number of words the tokenizer keeps
MAX_SEQ_LENGTH = 60      # pad/truncate all titles to this token length

# Paths (relative — script expects to be run from the keywordiq/ root)
RAW_DATA_PATH = os.path.join('data', 'raw', 'products_merged.csv')
PROCESSED_DIR = os.path.join('data', 'processed')


# ═════════════════════════════════════════════════════════════════════
# STEP 1: LOAD AND FIX ENCODING
# ═════════════════════════════════════════════════════════════════════

def load_and_fix_encoding(filepath: str) -> pd.DataFrame:
    """
    Load the merged CSV file with automatic encoding detection.
    Repairs mojibake characters caused by UTF-8/Latin-1 mismatches.

    Args:
        filepath: Path to the raw CSV file (products_merged.csv).

    Returns:
        pd.DataFrame with cleaned encoding in text columns.
    """
    # Try UTF-8 first, fall back to Latin-1
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')

    print(f"Loaded {len(df)} rows from {filepath}")

    def repair_encoding(text):
        """
        Fix mojibake by re-encoding Latin-1 → UTF-8, then strip any
        remaining non-ASCII bytes so downstream processing is clean.
        """
        if not isinstance(text, str):
            return str(text) if pd.notna(text) else ""
        try:
            # Step 1: decode the mistakenly-encoded bytes
            text = text.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        # Step 2: strip leftover non-ASCII characters entirely
        text = text.encode('ascii', errors='ignore').decode('ascii')
        return text

    # Apply encoding repair to the two text-heavy columns
    df['Product Title'] = df['Product Title'].apply(repair_encoding)

    if 'keyword_longtail' in df.columns:
        df['keyword_longtail'] = df['keyword_longtail'].apply(repair_encoding)
    else:
        # Create column with empty strings if it doesn't exist
        df['keyword_longtail'] = ""

    return df


# ═════════════════════════════════════════════════════════════════════
# STEP 2: CLEAN PRODUCT TITLES
# ═════════════════════════════════════════════════════════════════════

def clean_title(text: str) -> str:
    """
    Normalize a product title for tokenization.

    Steps:
        1. Lowercase
        2. Remove HTML tags
        3. Keep only letters, digits, spaces, and hyphens
        4. Collapse whitespace
        5. Strip edges

    Args:
        text: Raw product title string.

    Returns:
        Cleaned, lowercase product title.
    """
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)            # strip HTML tags
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)    # keep a-z, 0-9, space, hyphen
    text = re.sub(r'\s+', ' ', text)               # collapse multiple spaces
    return text.strip()


# ═════════════════════════════════════════════════════════════════════
# STEP 3: PARSE KEYWORD COLUMN
# ═════════════════════════════════════════════════════════════════════

def parse_keywords(value) -> list:
    """
    Parse the 'Keyword_Broad_Options' column, which stores Python lists
    as string literals (e.g. "['body wash', 'soap']"), NaN, or "[]".

    Args:
        value: Cell value — may be NaN, a list, or a string.

    Returns:
        A list of lowercase keyword strings with empty strings removed.
    """
    # NaN / None → empty list
    if isinstance(value, float) and pd.isna(value):
        return []

    # Already a list (rare, but possible after some pandas ops)
    if isinstance(value, list):
        return [str(k).strip().lower() for k in value if str(k).strip()]

    # Attempt ast.literal_eval on the string representation
    raw = str(value).strip()
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            keywords = [str(k).strip().lower() for k in parsed if str(k).strip()]
            return keywords
    except (ValueError, SyntaxError):
        pass

    # Manual fallback: strip brackets, split on commas, clean each piece
    try:
        inner = raw.strip('[]')
        parts = inner.split(',')
        keywords = [p.strip().strip("'\"").strip().lower() for p in parts]
        keywords = [k for k in keywords if k]
        return keywords
    except Exception:
        return []


# ═════════════════════════════════════════════════════════════════════
# STEP 4: FILTER BAD ROWS
# ═════════════════════════════════════════════════════════════════════

def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that cannot be used for training:
        - Empty keyword lists
        - Product titles shorter than 3 words
        - Duplicate product titles (keep first occurrence)

    Args:
        df: DataFrame after cleaning and keyword parsing.

    Returns:
        Filtered DataFrame.
    """
    original_count = len(df)

    # Drop rows with no keywords
    df = df[df['parsed_keywords'].apply(lambda x: len(x) > 0)].copy()

    # Drop rows where the cleaned title has fewer than 3 words
    df = df[df['Product Title'].apply(lambda t: len(str(t).split()) >= 3)].copy()

    # Deduplicate by Product Title, keeping first occurrence
    df = df.drop_duplicates(subset='Product Title', keep='first').copy()

    removed = original_count - len(df)
    print(f"Rows after filtering: {len(df)} (removed {removed} rows)")
    return df.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════
# STEP 5: BUILD KEYWORD VOCABULARY (Top-K only)
# ═════════════════════════════════════════════════════════════════════

def build_keyword_vocab(keyword_series: pd.Series,
                        top_k: int = 500) -> dict:
    """
    Construct a keyword vocabulary from the top-K most frequent keywords
    in the dataset.  Rare keywords that appear only a handful of times
    are discarded — the model cannot learn from them anyway, and keeping
    all ~40 k unique keywords creates a label matrix too large for RAM.

    Args:
        keyword_series: pandas Series where each element is a list of
                        keyword strings.
        top_k: Number of most-frequent keywords to retain (default 500).

    Returns:
        Dictionary with keys:
            - 'keyword_to_idx': {keyword_str: int_index}
            - 'idx_to_keyword': {int_index: keyword_str}
            - 'vocab_size': total number of unique keywords (== top_k)
    """
    # Flatten every keyword list into one big list
    all_keywords = [kw for kw_list in keyword_series for kw in kw_list]

    # Count frequencies for reporting
    counter = Counter(all_keywords)
    total_unique = len(counter)
    print(f"Total unique keywords found : {total_unique}")
    print(f"Keeping only top {top_k} most frequent keywords")

    # Keep only the top-K most frequent keywords
    top_keywords = [kw for kw, _count in counter.most_common(top_k)]
    cutoff_kw, cutoff_count = counter.most_common(top_k)[-1]
    print(f"Frequency cutoff: '{cutoff_kw}' appears {cutoff_count} times")

    # Build sorted vocabulary for reproducibility
    sorted_keywords = sorted(top_keywords)
    keyword_to_idx = {kw: idx for idx, kw in enumerate(sorted_keywords)}
    idx_to_keyword = {idx: kw for kw, idx in keyword_to_idx.items()}
    vocab_size = len(keyword_to_idx)

    print(f"Final keyword vocabulary size: {vocab_size}")
    print(f"Top 20 most common keywords: {[kw for kw, _ in counter.most_common(20)]}")

    return {
        'keyword_to_idx': keyword_to_idx,
        'idx_to_keyword': idx_to_keyword,
        'vocab_size': vocab_size,
    }


# ═════════════════════════════════════════════════════════════════════
# STEP 6: BUILD MULTI-HOT LABEL VECTORS
# ═════════════════════════════════════════════════════════════════════

def build_multi_hot_labels(keyword_series: pd.Series,
                           keyword_to_idx: dict,
                           vocab_size: int) -> np.ndarray:
    """
    Convert each row's keyword list into a fixed-length binary vector.

    For every product, the vector has 1.0 at positions corresponding to
    keywords that exist in the top-K vocabulary, and 0.0 elsewhere.
    Keywords outside the top-K are silently ignored.

    Args:
        keyword_series: Series of keyword lists (one list per product).
        keyword_to_idx: Mapping from keyword string to integer index
                        (only contains top-K keywords).
        vocab_size: Length of the output binary vector (== top-K).

    Returns:
        numpy array of shape (num_samples, vocab_size), dtype float32.
    """
    num_samples = len(keyword_series)
    y = np.zeros((num_samples, vocab_size), dtype=np.float32)

    for row_idx, kw_list in enumerate(keyword_series):
        for kw in kw_list:
            if kw in keyword_to_idx:   # only encode if in top-K vocab
                y[row_idx, keyword_to_idx[kw]] = 1.0

    return y


# ═════════════════════════════════════════════════════════════════════
# STEP 7: TOKENIZE TITLES
# ═════════════════════════════════════════════════════════════════════

def tokenize_titles(titles: pd.Series) -> tuple:
    """
    Fit a Keras Tokenizer on all product titles and convert titles to
    padded integer sequences.

    Args:
        titles: pandas Series of cleaned product title strings.

    Returns:
        Tuple of (padded_sequences, fitted_tokenizer).
        padded_sequences shape: (num_samples, MAX_SEQ_LENGTH).
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(
        num_words=MAX_VOCAB_SIZE,
        oov_token="<OOV>",        # handle unseen words at inference time
        lower=True,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )

    tokenizer.fit_on_texts(titles.tolist())
    sequences = tokenizer.texts_to_sequences(titles.tolist())
    padded = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH,
                           padding='post', truncating='post')

    return padded, tokenizer


# ═════════════════════════════════════════════════════════════════════
# STEP 8: BUILD LONGTAIL LOOKUP
# ═════════════════════════════════════════════════════════════════════

def build_longtail_lookup(df: pd.DataFrame) -> dict:
    """
    Create a dictionary mapping cleaned product titles to their
    corresponding longtail keyword phrase (for UI display only).

    Args:
        df: DataFrame containing 'Product Title' and 'keyword_longtail'.

    Returns:
        Dict {cleaned_title_str: longtail_phrase_str}.
    """
    lookup = {}
    for _, row in df.iterrows():
        title = str(row['Product Title']).strip()
        longtail = str(row.get('keyword_longtail', '')).strip()
        if longtail and longtail.lower() != 'nan':
            lookup[title] = longtail
    return lookup


# ═════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════

def main():
    """
    Execute the full preprocessing pipeline in order:
    Load → Clean → Parse → Filter → Vocab → Labels → Tokenize → Split → Save.
    """
    # Ensure output directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── Step 1: Load & encoding fix ──────────────────────────────────
    df = load_and_fix_encoding(RAW_DATA_PATH)
    df = df.head(50000).reset_index(drop=True)
    print(f"Trimmed to first 50,000 rows \u2014 vocab will be built from these only")
    total_loaded = len(df)

    # ── Step 2: Clean titles ─────────────────────────────────────────
    df['Product Title'] = df['Product Title'].apply(clean_title)

    # ── Step 3: Parse keyword column ─────────────────────────────────
    df['parsed_keywords'] = df['Keyword_Broad_Options'].apply(parse_keywords)

    # ── Step 4: Filter bad rows ──────────────────────────────────────
    df = filter_rows(df)
    samples_after_filter = len(df)

    # ── Step 5: Build keyword vocabulary ─────────────────────────────
    vocab_data = build_keyword_vocab(df['parsed_keywords'], top_k=500)
    keyword_to_idx = vocab_data['keyword_to_idx']
    vocab_size = vocab_data['vocab_size']

    vocab_path = os.path.join(PROCESSED_DIR, 'keyword_vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"Saved keyword vocabulary to {vocab_path}")

    # ── Step 6: Build multi-hot labels ───────────────────────────────
    y = build_multi_hot_labels(df['parsed_keywords'], keyword_to_idx, vocab_size)

    # ── Step 7: Tokenize titles ──────────────────────────────────────
    X, tokenizer = tokenize_titles(df['Product Title'])
    tokenizer_vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)

    tok_path = os.path.join(PROCESSED_DIR, 'tokenizer.pkl')
    with open(tok_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Saved tokenizer to {tok_path}")

    # ── Step 6b: Drop rows with NO top-K keyword match ───────────────
    # Some products only had rare keywords that fell outside the top-K;
    # they now have an all-zero label vector and add nothing to training.
    import sys
    valid_mask = y.sum(axis=1) > 0
    rows_before = len(y)
    y = y[valid_mask]
    X = X[valid_mask]
    # Also filter the DataFrame so longtail lookup stays in sync
    df = df[valid_mask].reset_index(drop=True)
    print(f"Rows with at least 1 Top-K keyword : {len(y)}")
    print(f"Rows dropped (no Top-K match)       : {rows_before - len(y)}")
    label_matrix_mb = sys.getsizeof(y) / (1024 * 1024)
    print(f"Label matrix size: {y.shape} = {label_matrix_mb:.1f} MB ✅")
    samples_after_filter = len(y)  # update for summary

    # ── Step 8: Build longtail lookup ────────────────────────────────
    longtail_lookup = build_longtail_lookup(df)
    lt_path = os.path.join(PROCESSED_DIR, 'longtail_lookup.pkl')
    with open(lt_path, 'wb') as f:
        pickle.dump(longtail_lookup, f)
    print(f"Saved longtail lookup ({len(longtail_lookup)} entries) to {lt_path}")

    # ── Step 9: Train / validation split ─────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        shuffle=True,
    )

    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_val.npy'), y_val)

    # ── Step 10: Print summary ───────────────────────────────────────
    print()
    print("╔══════════════════════════════════════╗")
    print("║        PREPROCESSING COMPLETE        ║")
    print("╠══════════════════════════════════════╣")
    print(f"║ Total samples loaded    : {total_loaded:<10} ║")
    print(f"║ Samples after filtering : {samples_after_filter:<10} ║")
    print(f"║ Keyword vocabulary size : {vocab_size:<10} ║")
    print(f"║ Tokenizer vocab size    : {tokenizer_vocab_size:<10} ║")
    print(f"║ Max sequence length     : {MAX_SEQ_LENGTH:<10} ║")
    print(f"║ Training samples        : {len(X_train):<10} ║")
    print(f"║ Validation samples      : {len(X_val):<10} ║")
    print(f"║ X_train shape           : {str(X_train.shape):<10} ║")
    print(f"║ y_train shape           : {str(y_train.shape):<10} ║")
    print("╚══════════════════════════════════════╝")
    print()
    print("All files saved to data/processed/")
    print("NEXT: Upload project to Google Drive and run colab_notebook.ipynb")


if __name__ == '__main__':
    main()
