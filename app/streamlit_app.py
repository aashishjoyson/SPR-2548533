"""
KeywordIQ — Streamlit Application
====================================
An AI-powered SEO keyword generator for Amazon product titles.
Designed for resellers and dropshippers who need to generate keywords
at scale without manual effort.

Features:
  - Single-title keyword prediction with confidence visualization
  - Batch CSV processing with downloadable results
  - Model performance insights and architecture explanation

Course: MAI417-3 Deep Learning | MSAIM | Christ University, Bangalore
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import streamlit as st
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KeywordIQ — AI Keyword Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Gradient title */
.main-title {
    background: linear-gradient(90deg, #1a73e8, #34a853, #fbbc05);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0;
}
/* Subtle card-style metric boxes */
div[data-testid="stMetric"] {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 12px 16px;
}
/* Keyword pill badges */
.kw-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    margin: 4px;
    font-size: 14px;
    color: white;
    font-weight: 500;
}
/* Primary button override */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #1a73e8, #34a853);
    border: none;
    color: white;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# LOAD MODEL ARTIFACTS (cached so they persist across reruns)
# ═════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_artifacts():
    """
    Load model artifacts once and cache in memory for the lifetime
    of the Streamlit server process.

    Returns:
        Tuple of (artifacts_tuple | None, error_message | None).
    """
    try:
        from utils.inference import load_artifacts
        return load_artifacts(), None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Unexpected error loading model: {str(e)}"


artifacts, error_message = get_artifacts()


# ═════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🔍 KeywordIQ")
    st.caption("Turn Amazon titles into SEO gold")
    st.divider()

    # ── Model status ─────────────────────────────────────────────────
    st.subheader("Model Status")
    model_exists = os.path.exists(os.path.join('model', 'keywordiq_model.h5'))
    if model_exists and artifacts is not None:
        st.success("✅ Model loaded and ready")
    elif model_exists and artifacts is None:
        st.error("❌ Model found but failed to load")
        if error_message:
            st.code(error_message, language="text")
    else:
        st.error("❌ Model not trained yet")
        st.info("Follow the training steps in README.md")

    st.divider()

    # ── Settings ─────────────────────────────────────────────────────
    st.subheader("Settings")
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.70,
        value=0.30,
        step=0.05,
        help="Lower = more keywords predicted. Higher = only very confident keywords shown.",
    )
    st.caption(f"Current: {threshold:.0%} minimum confidence")

    st.divider()

    # ── How it works ─────────────────────────────────────────────────
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        **Model Architecture:**
        - **Conv1D**: Detects local phrase patterns like "dry skin", "nail glue"
        - **BiLSTM**: Reads the full title for overall product category
        - **MLP Head**: Combines both to predict keyword probabilities
        - **Training**: Multi-label classification on Beauty & Health Amazon data

        **Why not text generation?**
        The model selects from a learned vocabulary of SEO keywords
        rather than generating words, making it faster and more reliable.
        """)


# ═════════════════════════════════════════════════════════════════════
# MAIN AREA — HEADER
# ═════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-title">KeywordIQ</p>', unsafe_allow_html=True)
st.markdown("**AI-Powered SEO Keyword Generator** for Amazon Product Titles")
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════
# THREE TABS
# ═════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["🔍 Single Title", "📦 Batch CSV", "📊 Model Insights"])


# ─────────────────────────────────────────────────────────────────────
# TAB 1 — SINGLE TITLE PREDICTION
# ─────────────────────────────────────────────────────────────────────

with tab1:
    st.header("Generate Keywords for One Product")
    st.caption("Paste any Amazon product title and get instant SEO keywords")

    # Example titles for demonstration
    EXAMPLE_TITLES = [
        "Aveeno Daily Moisturizing Body Wash, 12 Fl Oz",
        "Arctic Fox Vegan And Cruelty-Free Semi-Permanent Hair Color Dye 8 Fl Oz Virgin Pink",
        "Neutrogena Hydro Boost Water Gel Face Moisturizer with Hyaluronic Acid 1.7 Oz",
        "OGX Thick & Full Biotin & Collagen Shampoo 13 Fl Oz",
    ]

    example_choice = st.selectbox(
        "Load an example title or type your own below:",
        ["Type your own ↓"] + EXAMPLE_TITLES,
    )

    default_text = example_choice if example_choice != "Type your own ↓" else ""

    title_input = st.text_area(
        "Product Title:",
        value=default_text,
        height=100,
        placeholder="Paste any Amazon product title here...",
        help="The model works best with Beauty and Health category titles",
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        predict_btn = st.button(
            "🔍 Generate Keywords", type="primary", use_container_width=True,
        )

    if predict_btn:
        if not title_input.strip():
            st.warning("Please enter a product title first.")
        elif artifacts is None:
            st.error("Model not loaded. Please check the sidebar for status.")
        else:
            model, tokenizer, idx_to_keyword, vocab_size, longtail_lookup = artifacts

            with st.spinner("Analyzing title..."):
                from utils.inference import predict_keywords, clean_title
                predictions = predict_keywords(
                    title_input, model, tokenizer, idx_to_keyword, threshold,
                )

            if predictions:
                st.success(f"Found {len(predictions)} relevant keywords!")

                # ── Keyword badges ───────────────────────────────────
                st.subheader("Predicted Keywords:")
                badge_html = ""
                for keyword, confidence in predictions:
                    if confidence >= 0.6:
                        color = "#28a745"     # green — high confidence
                    elif confidence >= 0.4:
                        color = "#17a2b8"     # teal  — medium confidence
                    else:
                        color = "#6c757d"     # grey  — lower confidence
                    badge_html += (
                        f'<span class="kw-badge" style="background-color:{color};">'
                        f'{keyword} ({confidence:.0%})</span>'
                    )
                st.markdown(badge_html, unsafe_allow_html=True)

                st.divider()

                # ── Bar chart + copy box side by side ────────────────
                col_chart, col_copy = st.columns([3, 2])

                with col_chart:
                    import matplotlib.pyplot as plt
                    import matplotlib
                    matplotlib.use('Agg')

                    keywords_list = [kw for kw, _ in predictions]
                    scores_list   = [sc for _, sc in predictions]
                    colors = [
                        '#28a745' if s >= 0.6
                        else '#17a2b8' if s >= 0.4
                        else '#6c757d'
                        for s in scores_list
                    ]

                    fig, ax = plt.subplots(
                        figsize=(7, max(3, len(predictions) * 0.5)),
                    )
                    bars = ax.barh(
                        keywords_list[::-1], scores_list[::-1],
                        color=colors[::-1],
                    )
                    ax.set_xlabel('Confidence Score')
                    ax.set_title('Keyword Confidence Scores')
                    ax.set_xlim(0, 1)
                    ax.axvline(
                        x=threshold, color='red', linestyle='--',
                        alpha=0.7, label=f'Threshold ({threshold:.0%})',
                    )
                    ax.legend()

                    for bar, score in zip(bars, scores_list[::-1]):
                        ax.text(
                            score + 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            f'{score:.0%}', va='center', fontsize=9,
                        )

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col_copy:
                    st.subheader("📋 Copy Keywords")
                    comma_separated = ", ".join(
                        [kw for kw, _ in predictions],
                    )
                    st.text_area(
                        "Comma-separated (copy this):",
                        value=comma_separated, height=100,
                    )
                    bullet_format = "\n".join(
                        [f"• {kw} ({sc:.0%})" for kw, sc in predictions],
                    )
                    st.text_area(
                        "With confidence scores:",
                        value=bullet_format, height=150,
                    )

                    # Show longtail phrase if available in lookup
                    cleaned_input = clean_title(title_input)
                    longtail = longtail_lookup.get(cleaned_input, "")
                    if longtail:
                        st.info(f"💡 **Longtail phrase:** {longtail}")
            else:
                st.warning(
                    "No keywords predicted. Try lowering the confidence "
                    "threshold in the sidebar.",
                )


# ─────────────────────────────────────────────────────────────────────
# TAB 2 — BATCH CSV PROCESSING
# ─────────────────────────────────────────────────────────────────────

with tab2:
    st.header("Batch Process Multiple Titles")
    st.caption(
        "Upload a CSV with product titles — get keywords for all of them at once",
    )

    with st.expander("📋 CSV Format Requirements"):
        st.markdown("""
Your CSV file must have a column named exactly **`title`**
(or **`Product Title`**) containing your product titles.

**Example CSV structure:**
```
title
Aveeno Daily Moisturizing Body Wash 12 Fl Oz
Neutrogena Hydro Boost Water Gel 1.7 Oz
Arctic Fox Hair Color Dye Virgin Pink 8 Fl Oz
```
The app will add a `predicted_keywords` column and let you
download the result.
        """)

    uploaded_file = st.file_uploader(
        "Upload your CSV file:",
        type=['csv'],
        help="Must contain a 'title' or 'Product Title' column",
    )

    if uploaded_file is not None:
        # Read with encoding fallback
        try:
            df_upload = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            df_upload = pd.read_csv(uploaded_file, encoding='latin-1')

        # Auto-detect the title column
        title_col = None
        for possible_name in [
            'title', 'Title', 'Product Title', 'product_title', 'TITLE',
        ]:
            if possible_name in df_upload.columns:
                title_col = possible_name
                break

        if title_col is None:
            st.error(
                f"No title column found. Columns found: "
                f"{list(df_upload.columns)}",
            )
            st.info("Please rename your title column to 'title' and re-upload.")
        else:
            st.success(
                f"✅ Loaded {len(df_upload)} rows. "
                f"Title column: '{title_col}'",
            )

            # Preview
            st.subheader("Preview (first 5 rows):")
            st.dataframe(df_upload.head(), use_container_width=True)

            col_process, col_info = st.columns([1, 3])
            with col_process:
                process_btn = st.button(
                    "⚡ Process All Titles",
                    type="primary",
                    use_container_width=True,
                )
            with col_info:
                st.info(
                    f"Will process {len(df_upload)} titles using batch inference",
                )

            if process_btn:
                if artifacts is None:
                    st.error("Model not loaded.")
                else:
                    model, tokenizer, idx_to_keyword, vocab_size, longtail_lookup = artifacts
                    titles = (
                        df_upload[title_col].fillna("").astype(str).tolist()
                    )

                    progress_bar = st.progress(0)
                    status_text  = st.empty()
                    status_text.text("Running batch inference...")

                    from utils.inference import predict_batch

                    # Process in chunks to update the progress bar
                    CHUNK_SIZE  = 50
                    all_results = []
                    total       = len(titles)

                    for i in range(0, total, CHUNK_SIZE):
                        chunk = titles[i : i + CHUNK_SIZE]
                        chunk_results = predict_batch(
                            chunk, model, tokenizer, idx_to_keyword, threshold,
                        )
                        all_results.extend(chunk_results)
                        progress = min((i + CHUNK_SIZE) / total, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(
                            f"Processing... "
                            f"{min(i + CHUNK_SIZE, total)}/{total} titles",
                        )

                    progress_bar.progress(1.0)
                    status_text.text("✅ Complete!")

                    # Attach results to the dataframe
                    df_result = df_upload.copy()
                    df_result['predicted_keywords'] = all_results

                    st.subheader("Results:")
                    st.dataframe(
                        df_result[[title_col, 'predicted_keywords']],
                        use_container_width=True,
                    )

                    csv_output = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=csv_output,
                        file_name="keywordiq_results.csv",
                        mime="text/csv",
                        type="primary",
                    )

                    st.success(
                        f"Processed {len(df_upload)} titles successfully!",
                    )


# ─────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────

with tab3:
    st.header("Model Performance & Architecture")

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    history_path = os.path.join('training', 'history.pkl')
    curves_path  = os.path.join('training', 'training_curves.png')

    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            history = pickle.load(f)

        # ── Key metrics ──────────────────────────────────────────────
        best_epoch = int(np.argmax(history['val_auc'])) + 1
        best_auc   = max(history['val_auc'])
        best_acc   = history['val_binary_accuracy'][best_epoch - 1]
        total_epochs = len(history['loss'])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Val AUC",      f"{best_auc:.4f}")
        col2.metric("Best Val Accuracy",  f"{best_acc:.2%}")
        col3.metric("Best Epoch",         f"{best_epoch}/{total_epochs}")
        col4.metric(
            "Keyword Vocabulary",
            str(artifacts[3]) if artifacts else "N/A",
        )

        st.divider()

        # ── Training curves ──────────────────────────────────────────
        st.subheader("Training History")

        if os.path.exists(curves_path):
            st.image(
                curves_path,
                caption="Training vs Validation Metrics",
                use_container_width=True,
            )
        else:
            # Recreate from the history dict
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].plot(
                history['auc'], label='Train AUC',
                color='#1f77b4', linewidth=2,
            )
            axes[0].plot(
                history['val_auc'], label='Validation AUC',
                color='#ff7f0e', linewidth=2,
            )
            axes[0].axvline(
                x=best_epoch - 1, color='green', linestyle='--',
                alpha=0.7, label=f'Best Epoch ({best_epoch})',
            )
            axes[0].set_title('AUC Over Training Epochs', fontsize=14)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('AUC Score')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(
                history['loss'], label='Train Loss',
                color='#1f77b4', linewidth=2,
            )
            axes[1].plot(
                history['val_loss'], label='Validation Loss',
                color='#ff7f0e', linewidth=2,
            )
            axes[1].set_title('Loss Over Training Epochs', fontsize=14)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Binary Cross-Entropy Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()

        # ── Architecture table (for exam) ────────────────────────────
        st.subheader("Model Architecture")
        st.markdown("""
| Component | Details | Purpose |
|---|---|---|
| **Embedding Layer** | 10,000 vocab × 64 dims | Converts words to dense vectors |
| **Conv1D (128 filters, k=3)** | Trigram detector | Captures 3-word local patterns |
| **Conv1D (64 filters, k=2)** | Bigram detector | Captures 2-word local patterns |
| **GlobalMaxPooling1D** | Reduces to 64-dim | Extracts strongest signal |
| **Bidirectional LSTM (64)** | Forward + Backward | Full title sequential context |
| **Concatenate** | 64 + 128 = 192-dim | Merges CNN and LSTM features |
| **Dense (256) + Dropout(0.4)** | MLP + Regularization | Deep classification |
| **Dense (128) + Dropout(0.3)** | MLP + Regularization | Refinement layer |
| **Dense (vocab_size, sigmoid)** | Output layer | Independent keyword probabilities |

**Loss:** Binary Cross-Entropy (each keyword is independent yes/no)
**Optimizer:** Adam (lr=0.001 with ReduceLROnPlateau)
**Regularization:** Dropout (0.4, 0.3) + L2 weight decay (1e-4)
        """)
    else:
        st.info("📊 Training history not found yet.")
        st.markdown("""
**To see insights here:**
1. Complete the model training in Google Colab
2. Download `history.pkl` from Colab
3. Place it in the `training/` folder
4. Restart the Streamlit app
        """)

        # Show architecture even without training history
        st.subheader("Planned Model Architecture")
        st.markdown("""
**KeywordIQ uses a Parallel CNN + BiLSTM + MLP architecture:**
- **Input:** Product title as integer token sequence (length 60)
- **Shared Embedding:** Maps tokens to 64-dimensional vectors
- **CNN Branch:** Two Conv1D layers detect short keyword phrases
- **BiLSTM Branch:** Reads full title for product category context
- **Merge:** Concatenation → 192-dimensional combined vector
- **MLP Head:** 256 → 128 → vocab_size with Dropout regularization
- **Output:** Sigmoid probabilities for each keyword in vocabulary
        """)
