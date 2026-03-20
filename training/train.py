"""
KeywordIQ — Model Training Script
===================================
Defines and trains a parallel CNN + BiLSTM multi-label classification
model for predicting SEO keywords from Amazon product titles.

Designed to run on Google Colab with T4 GPU acceleration.
Loads preprocessed .npy and .pkl files produced by preprocess.py,
builds the model, trains with early stopping, and saves the best
weights to model/keywordiq_model.h5.

Architecture (Parallel CNN + BiLSTM + MLP):
  - CNN Branch : Conv1D layers detect local n-gram patterns
                 ("dry skin", "nail glue", "body wash")
  - BiLSTM Branch : Bidirectional LSTM reads the full title for
                    overall product category context
  - Shared Embedding : Both branches share one embedding layer
  - MLP Head : Dense + Dropout layers map the merged representation
               to per-keyword sigmoid probabilities
  - Output : Binary vector — one sigmoid neuron per keyword

Course: MAI417-3 Deep Learning | MSAIM | Christ University, Bangalore
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Bidirectional, LSTM, Concatenate, Dense, Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from tensorflow.keras.regularizers import l2

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS — keep in sync with training/preprocess.py
# ─────────────────────────────────────────────────────────────────────
MAX_VOCAB_SIZE   = 10000   # tokenizer vocabulary cap
MAX_SEQ_LENGTH   = 60      # padded title length in tokens
EMBEDDING_DIM    = 64      # word-vector dimensionality
LSTM_UNITS       = 64      # per-direction LSTM hidden size (BiLSTM = 128)
CNN_FILTERS_1    = 128     # first Conv1D filter count (trigrams)
CNN_FILTERS_2    = 64      # second Conv1D filter count (bigrams)
DENSE_UNITS_1    = 256     # first MLP hidden layer width
DENSE_UNITS_2    = 128     # second MLP hidden layer width
DROPOUT_RATE_1   = 0.4     # dropout after first dense layer
DROPOUT_RATE_2   = 0.3     # dropout after second dense layer
LEARNING_RATE    = 0.001   # Adam initial learning rate
BATCH_SIZE       = 64      # mini-batch size
MAX_EPOCHS       = 30      # upper bound on training epochs
PATIENCE         = 5       # EarlyStopping patience (epochs w/o improvement)
MODEL_SAVE_PATH  = os.path.join('model', 'keywordiq_model.h5')


# ═════════════════════════════════════════════════════════════════════
# LOAD PREPROCESSED DATA
# ═════════════════════════════════════════════════════════════════════

def load_data() -> tuple:
    """
    Load training arrays and keyword vocabulary produced by preprocess.py.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val, vocab_size).
    """
    processed = os.path.join('data', 'processed')

    X_train = np.load(os.path.join(processed, 'X_train.npy'))
    X_val   = np.load(os.path.join(processed, 'X_val.npy'))
    y_train = np.load(os.path.join(processed, 'y_train.npy'))
    y_val   = np.load(os.path.join(processed, 'y_val.npy'))

    with open(os.path.join(processed, 'keyword_vocab.pkl'), 'rb') as f:
        vocab_data = pickle.load(f)
    vocab_size = vocab_data['vocab_size']

    print("─── Data loaded successfully ───")
    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}   y_val   : {y_val.shape}")
    print(f"  Keyword vocabulary size : {vocab_size}")
    return X_train, X_val, y_train, y_val, vocab_size


# ═════════════════════════════════════════════════════════════════════
# BUILD MODEL
# ═════════════════════════════════════════════════════════════════════

def build_keywordiq_model(vocab_size: int,
                          max_seq_len: int    = MAX_SEQ_LENGTH,
                          embedding_dim: int  = EMBEDDING_DIM,
                          lstm_units: int     = LSTM_UNITS,
                          cnn_filters_1: int  = CNN_FILTERS_1,
                          cnn_filters_2: int  = CNN_FILTERS_2,
                          dense_units_1: int  = DENSE_UNITS_1,
                          dense_units_2: int  = DENSE_UNITS_2,
                          dropout_rate_1: float = DROPOUT_RATE_1,
                          dropout_rate_2: float = DROPOUT_RATE_2) -> Model:
    """
    Build the parallel CNN + BiLSTM multi-label classification model
    using the Keras Functional API.

    Architecture
    ------------
    Input → Embedding (shared)
              ├── Conv1D(128, k=3) → Conv1D(64, k=2) → GlobalMaxPool  [CNN branch]
              └── Bidirectional LSTM(64)                                [LSTM branch]
              ──> Concatenate → Dense(256) → Dropout → Dense(128) →
                  Dropout → Dense(vocab_size, sigmoid)

    Args:
        vocab_size:     Number of unique keywords (output neurons).
        max_seq_len:    Padded sequence length (default 60).
        embedding_dim:  Word-vector dimension (default 64).
        lstm_units:     LSTM hidden units per direction (default 64).
        cnn_filters_1:  Filters for the trigram Conv1D layer (default 128).
        cnn_filters_2:  Filters for the bigram Conv1D layer (default 64).
        dense_units_1:  Width of first MLP layer (default 256).
        dense_units_2:  Width of second MLP layer (default 128).
        dropout_rate_1: Dropout probability after first dense (default 0.4).
        dropout_rate_2: Dropout probability after second dense (default 0.3).

    Returns:
        Compiled-ready Keras Model (not yet compiled).
    """
    # ── Input ────────────────────────────────────────────────────────
    inputs = Input(shape=(max_seq_len,), name='title_input')

    # ── Shared Embedding ─────────────────────────────────────────────
    # +1 accounts for the <OOV> token index
    embedding = Embedding(
        input_dim=MAX_VOCAB_SIZE + 1,
        output_dim=embedding_dim,
        input_length=max_seq_len,
        trainable=True,               # learn embeddings from scratch
        name='shared_embedding',
    )(inputs)
    # Shape: (batch, 60, 64)

    # ── CNN Branch — local phrase pattern detection ──────────────────
    # First Conv1D: trigram detector (kernel_size=3)
    # A 3-word sliding window over the title can learn patterns like
    # "dry skin body", "body wash for", "semi permanent hair"
    cnn = Conv1D(
        filters=cnn_filters_1,
        kernel_size=3,
        activation='relu',
        padding='same',
        name='conv1d_trigram',
    )(embedding)

    # Second Conv1D: bigram detector (kernel_size=2)
    # Operates on the trigram feature maps; effectively captures
    # overlapping 2-word sub-patterns within the trigram context
    cnn = Conv1D(
        filters=cnn_filters_2,
        kernel_size=2,
        activation='relu',
        padding='same',
        name='conv1d_bigram',
    )(cnn)

    # GlobalMaxPooling extracts the single strongest activation per
    # filter across all token positions → fixed-length vector
    cnn_out = GlobalMaxPooling1D(name='global_max_pool')(cnn)
    # Shape: (batch, 64)

    # ── BiLSTM Branch — full sequential context ──────────────────────
    # Reads the title in both directions simultaneously so that later
    # tokens can inform the meaning of earlier ones and vice versa.
    # return_sequences=False keeps only the final hidden state.
    lstm_out = Bidirectional(
        LSTM(lstm_units, return_sequences=False),
        name='bilstm',
    )(embedding)
    # Shape: (batch, 128) — 64 forward + 64 backward

    # ── Merge ────────────────────────────────────────────────────────
    merged = Concatenate(name='merge_cnn_lstm')([cnn_out, lstm_out])
    # Shape: (batch, 192) — 64 CNN + 128 BiLSTM

    # ── MLP Classification Head ──────────────────────────────────────
    x = Dense(
        dense_units_1, activation='relu',
        kernel_regularizer=l2(1e-4),   # L2 weight penalty (exam concept)
        name='dense_1',
    )(merged)

    # Dropout randomly zeroes 40 % of activations during training,
    # preventing co-adaptation of hidden units
    x = Dropout(dropout_rate_1, name='dropout_1')(x)

    x = Dense(
        dense_units_2, activation='relu',
        kernel_regularizer=l2(1e-4),
        name='dense_2',
    )(x)

    x = Dropout(dropout_rate_2, name='dropout_2')(x)

    # ── Output ───────────────────────────────────────────────────────
    # One neuron per keyword, each outputting an independent probability.
    # Sigmoid (NOT softmax) because multiple keywords can be active
    # simultaneously; softmax would force probabilities to sum to 1.
    outputs = Dense(
        vocab_size, activation='sigmoid',
        name='keyword_output',
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name='KeywordIQ')
    return model


# ═════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ═════════════════════════════════════════════════════════════════════

def main():
    """
    Full training pipeline: load data → build model → compile →
    fit with callbacks → save history → print summary.
    """
    # Ensure model output directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val, vocab_size = load_data()

    # ── Build model ──────────────────────────────────────────────────
    model = build_keywordiq_model(vocab_size)

    # ── Compile ──────────────────────────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'binary_accuracy',
            AUC(name='auc', multi_label=True),
        ],
    )

    # Print the full architecture table (useful for exam report)
    model.summary()

    # ── Callbacks ────────────────────────────────────────────────────
    callbacks = [
        # Stop training when validation AUC stops improving
        EarlyStopping(
            monitor='val_auc',
            patience=PATIENCE,
            restore_best_weights=True,   # go back to best epoch weights
            mode='max',                   # higher AUC is better
            verbose=1,
        ),
        # Save the model only when val_auc beats the previous best
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
        # Halve the learning rate if val_loss does not decrease for 3 epochs
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── Train ────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  Starting KeywordIQ training...")
    print(f"  Training samples   : {len(X_train)}")
    print(f"  Validation samples : {len(X_val)}")
    print(f"  Keyword vocab size : {vocab_size}")
    print(f"  Max epochs         : {MAX_EPOCHS}")
    print(f"  Batch size         : {BATCH_SIZE}")
    print(f"  Early stop patience: {PATIENCE}")
    print("=" * 55)
    print()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save training history for plotting in the notebook / app ─────
    history_path = os.path.join('training', 'history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to {history_path}")

    # ── Print final results ──────────────────────────────────────────
    best_epoch = int(np.argmax(history.history['val_auc'])) + 1
    best_val_auc = max(history.history['val_auc'])
    best_val_acc = history.history['val_binary_accuracy'][best_epoch - 1]

    print()
    print("╔══════════════════════════════════════╗")
    print("║         TRAINING COMPLETE            ║")
    print("╠══════════════════════════════════════╣")
    print(f"║ Best Epoch          : {best_epoch:<14} ║")
    print(f"║ Best Val AUC        : {best_val_auc:<14.4f} ║")
    print(f"║ Best Val Accuracy   : {best_val_acc * 100:<13.2f}% ║")
    print(f"║ Model saved to      : {MODEL_SAVE_PATH:<14} ║")
    print("╚══════════════════════════════════════╝")


if __name__ == '__main__':
    main()
