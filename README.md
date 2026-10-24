# SentimentScope 🎯

A production-grade, beginner-friendly Sentiment Analysis web application and machine learning benchmark comparing classical NLP (TF-IDF + Logistic Regression) against state-of-the-art Transformer architectures (DistilBERT) on the Stanford Sentiment Treebank (SST-2) dataset.

---

## 🌟 Key Highlights

- **Rigorous ML Methodology:** Completely honest evaluation on the labelled SST-2 validation split (872 sentences) with strict data-leakage prevention (TF-IDF fitted solely on training split).
- **Dual Model Architecture:**
  - **Classical Baseline:** Fast, interpretable TF-IDF Vectorizer with unigram & bigram features paired with Logistic Regression.
  - **Transformer:** DistilBERT pre-trained and fine-tuned for sequence classification on SST-2.
- **Interactive Streamlit Web Dashboard:**
  - **Live Sentiment Analyzer:** Real-time text analysis with confidence scores, probability gauge, and breakdown.
  - **Model Comparison:** Side-by-side performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) and confusion matrices.
  - **Dataset Explorer:** Search and filter SST-2 train/validation sentences.
  - **Methodology & Documentation:** Clear, educational explanations of NLP concepts.

---

## 📁 Project Structure

```
SentimentScope/
├── .gitignore               # Git ignore for venv, data, and model checkpoints
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── REPORT.md                # Comprehensive academic & technical report
├── data/
│   ├── raw/                 # Raw data files
│   └── processed/           # Processed SST-2 parquet caches
├── models/
│   ├── baseline/            # TF-IDF vectorizer, logistic regression model & metrics
│   └── transformer/         # Transformer evaluation metrics & configs
├── src/
│   ├── __init__.py          # Package initialization
│   ├── preprocessor.py      # Text normalization and cleaning
│   ├── data_loader.py       # SST-2 dataset fetcher & cache manager
│   ├── baseline_model.py    # TF-IDF + Logistic Regression training & inference
│   ├── transformer_model.py # DistilBERT evaluation & inference
│   └── utils.py             # Metrics calculation and plotting helpers
├── app/
│   ├── __init__.py
│   ├── main.py              # Main Streamlit web application
│   └── components/          # Modular Streamlit page views
└── tests/
    └── test_pipeline.py     # Automated unit tests
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (Recommended Python 3.12)
- Windows PowerShell / Linux / macOS

### 2. Environment Setup
Clone or navigate to the repository folder:
```powershell
cd C:\Users\pavan\SentimentScope
```

Create and activate a virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:
```powershell
pip install -r requirements.txt
```

### 3. Train & Evaluate the Baseline Model
Run the baseline training script:
```powershell
python -m src.baseline_model
```
This downloads SST-2 via Hugging Face datasets, fits the TF-IDF vectorizer solely on the 67,349 training sentences, evaluates strictly on the 872 validation sentences, and saves the models and metrics to `models/baseline/`.

### 4. Evaluate Transformer (DistilBERT)
```powershell
python -m src.transformer_model
```

### 5. Launch the Streamlit Web Application
```powershell
streamlit run app/main.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🧪 Testing
Run the automated test suite to verify preprocessing, metrics calculation, and model persistence:
```powershell
pytest tests/
```

---

## 📊 Dataset & Splitting Details

- **Dataset:** Stanford Sentiment Treebank (SST-2) via [Hugging Face `stanfordnlp/sst2`](https://huggingface.co/datasets/stanfordnlp/sst2)
- **Train Set:** 67,349 labelled movie reviews (0: Negative, 1: Positive)
- **Validation Set:** 872 labelled movie reviews used for genuine benchmarking
- **Data Leakage Prevention:** Vectorizer vocabularies and IDF scales are strictly learned on the training split and transformed on validation data.
