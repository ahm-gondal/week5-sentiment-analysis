# Week 5 — NLP & Sentiment Analysis Dashboard

**AI & ML Internship — Task 5**

## Project Overview

Teach a machine to read Amazon customer reviews and predict whether each is **positive** or **negative**, then wrap it in an interactive dashboard. **Part 1** (notebook) cleans raw text, converts it to numbers, trains and compares classifiers, and saves the best one. **Part 2** is a **Streamlit** app where a user types any review and instantly gets the predicted sentiment with a confidence score. NLP powers search, chatbots, and customer-feedback tools — one of the most in-demand skills in the industry.

## Dataset

- **Name:** Amazon Product Reviews (Alexa)
- **Size:** ~3,150 real reviews; text column `verified_reviews`, sentiment label `feedback` (1 = positive, 0 = negative)
- **Note:** heavily imbalanced (mostly positive) — handled with macro-F1 evaluation and balanced class weights

## Approach (Part 1)

1. **Inspect + class distribution** — confirmed the positive/negative imbalance.
2. **Text cleaning** (`clean_text`) — lowercase, strip HTML/URLs/non-letters, remove stopwords (scikit-learn's list), drop very short tokens. The *same* function is reused in the app so training and serving are consistent.
3. **Word clouds** — separate clouds for positive vs negative reviews show clearly different vocabularies.
4. **Vectorization — TF-IDF** (unigrams + bigrams, top 5,000 terms). Chosen because it up-weights distinctive words (*disappointed*, *refund*, *love*) and down-weights common ones, and pairs strongly with linear models.
5. **Two models** — Logistic Regression (balanced) and Multinomial Naive Bayes; 80/20 stratified split.
6. **Evaluation** — accuracy, precision, recall, macro-F1, and a confusion matrix per model.
7. **Save** — the best model + its vectorizer are saved together as `sentiment_model.joblib`.

## Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| **Logistic Regression** (chosen) | **0.931** | **0.803** |
| Multinomial Naive Bayes | 0.925 | 0.501 |

**Why Logistic Regression, not Naive Bayes?** They have nearly the same accuracy, but Naive Bayes' macro-F1 is only 0.50 — it mostly predicts the majority (positive) class and fails on negatives. Logistic Regression's much higher macro-F1 (0.80) shows it actually detects negative reviews. This is a textbook example of why accuracy is misleading on imbalanced data. Live sanity check: *"I absolutely love it"* → Positive (93%), *"Terrible, I want a refund"* → Negative (94%).

## Part 2 — Streamlit Dashboard

Three pages via the sidebar:

- **Home** — project + dataset intro.
- **Data Overview** — class-distribution chart and the two word clouds.
- **Sentiment Predictor** — type a review, click Predict, get the sentiment + confidence. It loads `sentiment_model.joblib` and applies the exact same `clean_text` cleaning before predicting.

### Run it locally

```bash
git clone https://github.com/ahm-gondal/week5-sentiment-analysis.git
cd week5-sentiment-analysis
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Repository Structure

```
├── app.py                       # Streamlit dashboard (3 pages)
├── sentiment_model.joblib       # Saved model + TF-IDF vectorizer
├── class_distribution.png       # used by the Data Overview page
├── wordcloud_positive.png
├── wordcloud_negative.png
├── notebooks/
│   └── week5_nlp.ipynb          # Part 1 — executed, all outputs visible
├── charts/                      # confusion matrices, model comparison, word clouds
├── data/                        # dataset + notes
├── README.md
└── requirements.txt
```

## Tools

Python · pandas · NumPy · scikit-learn (TF-IDF, LogisticRegression, MultinomialNB) · WordCloud · Matplotlib · Seaborn · Streamlit · joblib
