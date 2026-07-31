"""
Week 5 — Sentiment Analysis Streamlit Dashboard
Run with:  streamlit run app.py
Three pages (sidebar): Home · Data Overview · Sentiment Predictor
"""

import os
import re
import joblib
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="Amazon Review Sentiment", page_icon="💬", layout="centered")

# ---------------------------------------------------------------------------
# Text cleaning — IDENTICAL to the function used in the training notebook
# ---------------------------------------------------------------------------
STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Load the saved model + vectorizer once (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    for p in ["sentiment_model.joblib", "models/sentiment_model.joblib"]:
        if os.path.exists(p):
            return joblib.load(p)
    return None


bundle = load_bundle()


def img_path(name):
    for p in (name, os.path.join("charts", name), os.path.join("static", name)):
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("💬 Sentiment App")
page = st.sidebar.radio("Navigate", ["Home", "Data Overview", "Sentiment Predictor"])

# ============================ HOME ============================
if page == "Home":
    st.title("💬 Amazon Review Sentiment Analysis")
    st.markdown(
        """
        Welcome! This app was built for **Week 5** of the AI & ML internship.

        It uses **Natural Language Processing (NLP)** to read a customer review and predict
        whether it is **positive** or **negative**.

        **The pipeline behind it:**
        1. Clean the raw review text (lowercase, remove noise and stopwords).
        2. Convert the text to numbers with a **TF-IDF** vectorizer.
        3. Classify with a **Logistic Regression** model trained on real Amazon reviews.

        **Dataset:** ~3,150 real Amazon (Alexa) product reviews, each labelled positive or negative.

        Use the sidebar to explore the data or try the live **Sentiment Predictor**.
        """
    )
    if bundle:
        st.success(f"Model loaded: **{bundle.get('best_name', 'classifier')}** + TF-IDF vectorizer ✅")
    else:
        st.warning("Model file `sentiment_model.joblib` not found in the app folder.")

# ======================= DATA OVERVIEW =======================
elif page == "Data Overview":
    st.title("📊 Data Overview")

    st.subheader("Class Distribution")
    st.write(
        "The dataset is **imbalanced** — most reviews are positive. "
        "That's why the model was evaluated with macro-F1 (not plain accuracy) and trained "
        "with balanced class weights."
    )
    cd = img_path("class_distribution.png")
    if cd:
        st.image(cd, use_column_width=True)

    st.subheader("Word Clouds")
    st.write("The most frequent words in positive vs negative reviews — the vocabularies are clearly different.")
    col1, col2 = st.columns(2)
    pos = img_path("wordcloud_positive.png")
    neg = img_path("wordcloud_negative.png")
    with col1:
        st.markdown("**Positive reviews**")
        if pos:
            st.image(pos, use_column_width=True)
    with col2:
        st.markdown("**Negative reviews**")
        if neg:
            st.image(neg, use_column_width=True)

    combined = img_path("wordclouds.png")
    if not (pos and neg) and combined:
        st.image(combined, use_column_width=True)

# ===================== SENTIMENT PREDICTOR =====================
elif page == "Sentiment Predictor":
    st.title("🔮 Sentiment Predictor")
    st.write("Type any product review below and click **Predict** to see its sentiment and the model's confidence.")

    review = st.text_area("Your review", height=150,
                          placeholder="e.g. This speaker is amazing, the sound quality blew me away!")

    if st.button("Predict Sentiment", type="primary"):
        if bundle is None:
            st.error("Model not loaded — make sure sentiment_model.joblib is in the app folder.")
        elif not review.strip():
            st.warning("Please type a review first.")
        else:
            model = bundle["model"]
            vectorizer = bundle["vectorizer"]
            cleaned = clean_text(review)
            vec = vectorizer.transform([cleaned])
            label = int(model.predict(vec)[0])
            confidence = float(model.predict_proba(vec)[0].max())

            if label == 1:
                st.success(f"### ✅ Positive\nConfidence: **{confidence:.1%}**")
            else:
                st.error(f"### ❌ Negative\nConfidence: **{confidence:.1%}**")

            with st.expander("See the cleaned text the model actually used"):
                st.code(cleaned or "(empty after cleaning)")

st.sidebar.markdown("---")
st.sidebar.caption("Week 5 · NLP & Sentiment Analysis · AI & ML Internship")
