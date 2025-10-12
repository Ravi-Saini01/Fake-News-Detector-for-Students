# 🧠 Fake News Detector for Students

**Fake News Detector for Students** is a web application that helps students identify misinformation online. The app uses AI to detect whether a news article is **real or fake** and provides a **concise summary** of the content, helping students quickly understand the credibility of online news.

---

## 🔗 Live Demo

https://fake-news-detector-for-students1.streamlit.app/


---

## 📝 Features

- 📰 Detect fake or real news articles  
- 🔗 Accept **article text** or **news URL** as input  
- 🧠 Summarize long articles automatically using AI  
- 💾 Trained on 20,000+ real-world articles for better accuracy  
- ✅ Interactive, user-friendly interface built with **Streamlit**  

---

## 📦 Technology Stack

- **Python**  
- **Streamlit** for web interface  
- **scikit-learn**: TF-IDF + PassiveAggressiveClassifier for fake news detection  
- **Transformers (Hugging Face BART)** for summarization  
- **newspaper3k** & **BeautifulSoup** for web article extraction  
- **Pandas & NumPy** for data processing  

---

## Install dependencies:
pip install -r requirements.txt
## python -m streamlit run app.py

## Usage

Open the app in your browser.

Choose input type: Paste article text or enter a news URL.

Click Analyze.

The app will display:

Prediction: Fake or Real

Summary: Concise summary of the article

## Future Enhancements

Add confidence score for prediction (0–100%)

Show source credibility of the URL

Visual analytics (charts, keyword highlights, sentiment analysis)

Multi-language support

## References

Fake and Real News Dataset on Kaggle

newspaper3k documentation

Hugging Face Transformers

Streamlit documentation
