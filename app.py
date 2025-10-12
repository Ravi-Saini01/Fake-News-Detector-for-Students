import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from transformers import pipeline
from newspaper import Article
import requests
from bs4 import BeautifulSoup
import pickle
import os

# ---------------------
# Streamlit Setup
# ---------------------
st.set_page_config(page_title="Fake News Detector for Students 🧠", page_icon="📰", layout="wide")

st.title("🧠 Fake News Detector for Students")
st.markdown("### Detect misinformation and get concise summaries from credible AI models!")

# ---------------------
# Data Loading and Model Training
# ---------------------
@st.cache_resource
def load_fake_news_model():
    # Check if trained model exists
    if os.path.exists("fake_news_model.pkl"):
        with open("fake_news_model.pkl", "rb") as f:
            model, vectorizer, acc = pickle.load(f)
        return model, vectorizer, acc

    # Load Kaggle dataset
    st.write("📥 Loading dataset and training model... (only once)")
    fake_df = pd.read_csv("Fake.csv")
    true_df = pd.read_csv("True.csv")

    # Label the data
    fake_df["label"] = "FAKE"
    true_df["label"] = "REAL"

    # Combine and shuffle
    df = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train PassiveAggressive Classifier
    model = PassiveAggressiveClassifier(max_iter=100)
    model.fit(X_train_tfidf, y_train)

    # Evaluate accuracy
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    # Save model
    with open("fake_news_model.pkl", "wb") as f:
        pickle.dump((model, vectorizer, acc), f)

    return model, vectorizer, acc


model, vectorizer, acc = load_fake_news_model()
st.success(f"✅ Model ready! Trained with accuracy: **{acc*100:.2f}%**")

# ---------------------
# Summarization Model
# ---------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# ---------------------
# Helper Functions
# ---------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()

def predict_fake_news(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

def fetch_article_from_url(url):
    """Try to extract text using newspaper3k and fallback to BeautifulSoup."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        if not text.strip():
            raise ValueError("Empty article")
        return text
    except Exception:
        try:
            page = requests.get(url, timeout=10)
            soup = BeautifulSoup(page.content, "html.parser")
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text() for p in paragraphs)
            return text if len(text) > 100 else None
        except Exception:
            return None

# ---------------------
# User Input Section
# ---------------------
st.subheader("📰 Analyze a News Article")

input_type = st.radio("Choose input type:", ["Paste Article Text", "Enter Article URL"])

article_text = ""

if input_type == "Paste Article Text":
    article_text = st.text_area("Paste your article text here:", height=200)
else:
    url = st.text_input("Enter the article URL:")
    if url:
        with st.spinner("Fetching article content... 🔍"):
            article_text = fetch_article_from_url(url)
        if article_text:
            st.success("✅ Article fetched successfully!")
            with st.expander("📄 View extracted text"):
                st.write(article_text[:2000] + ("..." if len(article_text) > 2000 else ""))
        else:
            st.error("⚠️ Unable to extract content from the provided URL.")

# ---------------------
# Prediction Section
# ---------------------
if st.button("🔍 Analyze"):
    if not article_text.strip():
        st.warning("⚠️ Please enter article text or provide a valid URL.")
    else:
        with st.spinner("Analyzing article... please wait ⏳"):
            label = predict_fake_news(article_text)
            summary = summarizer(article_text, max_length=100, min_length=25, do_sample=False)[0]["summary_text"]

        if label == "FAKE":
            st.error("🚨 This news appears to be **FAKE** or unreliable.")
        else:
            st.success("✅ This news appears to be **REAL** and credible.")

        st.markdown("### 📝 Summary of the Article:")
        st.info(summary)

# ---------------------
# About Section
# ---------------------
st.markdown("---")
st.markdown("""
### 📘 About this App
**Fake News Detector for Students** uses AI to help students verify online information.

**Features**
- 🧠 Detects whether news is real or fake  
- 🔗 Accepts article text or URL input  
- 🗞️ Automatically summarizes long articles  
- 💾 Trained on 20,000+ real-world articles  

  
""")
