# SentimentScope: Sentiment Analysis Technical & Academic Report

**Project Title:** SentimentScope: Comparative Sentiment Classification on the Stanford Sentiment Treebank (SST-2)  
**Author / Workspace:** C:\Users\pavan\SentimentScope  
**Dataset:** Stanford Sentiment Treebank (SST-2) via Hugging Face (`stanfordnlp/sst2`)  
**Repository Branch:** `main`  

---

## 1. Executive Summary & Problem Statement

Sentiment Analysis is the automated computational extraction and classification of emotional valence, opinion, and attitude expressed in written natural language. With exponential growth in user-generated content—such as product reviews, social media discourse, movie evaluations, and customer feedback—enterprises require automated, high-precision tools to gauge public sentiment.

In this project, we design, implement, and rigorously benchmark **SentimentScope**, an end-to-end sentiment classification system. We systematically compare two distinct machine learning paradigms:
1. **Classical Linear NLP Baseline:** Unigram & Bigram Term Frequency-Inverse Document Frequency (TF-IDF) feature representations coupled with L2-regularized Logistic Regression.
2. **Modern Transformer Architecture:** DistilBERT (Bidirectional Encoder Representations from Transformers via Knowledge Distillation) fine-tuned for sequence classification on SST-2.

The key engineering and academic goals of SentimentScope are:
- **Zero Fabrication:** Compute and report genuine metrics strictly on the standard validation split.
- **Data Leakage Elimination:** Isolate all vocabulary extraction, document frequency computation, and statistical fitting to the training partition.
- **Production-Ready Web Application:** Deploy an intuitive, responsive Streamlit web application providing live sentiment analysis, side-by-side metric visualizations, and dataset exploration.

---

## 2. Dataset Description: Stanford Sentiment Treebank (SST-2)

The Stanford Sentiment Treebank v2 (SST-2) is one of the foundational binary classification benchmarks in the General Language Understanding Evaluation (GLUE) benchmark suite. Originating from Rotten Tomatoes movie review snippets annotated by human judges, SST-2 captures complex linguistic phenomena including irony, negation, and idiom.

### Partitioning & Split Hygiene
| Split | Total Samples | Positive Class (1) | Negative Class (0) | Purpose in SentimentScope |
| :--- | :--- | :--- | :--- | :--- |
| **Train Split** | 67,349 | ~37,573 (55.8%) | ~29,776 (44.2%) | Sole training data for TF-IDF and classifiers |
| **Validation Split** | 872 | 444 (50.9%) | 428 (49.1%) | Sole evaluation benchmark for all reported metrics |
| **Test Split** | 1,821 | Unlabelled | Unlabelled | GLUE leaderboard submission split (held-out) |

### Mitigation of Data Leakage
Data leakage occurs when information from outside the training partition influences the model training process. In NLP, this commonly happens if a vectorizer (like TF-IDF) computes document frequencies across both training and validation sets. In SentimentScope:
- `TfidfVectorizer.fit_transform()` is strictly invoked on `train_df["sentence"]`.
- The validation data `val_df["sentence"]` is strictly transformed via `TfidfVectorizer.transform()`.
- No validation sentence or label is ever seen by the feature extractor during fitting.

---

## 3. Data Preprocessing & Cleaning Pipeline

Real-world user text frequently contains formatting anomalies, HTML entities, and URLs that add noise to statistical representations. Our preprocessing pipeline (`src/preprocessor.py`) applies:
1. **HTML Entity Decoding:** Unescapes entities like `&amp;` to `&`, `&quot;` to `"`, avoiding fragmented tokens.
2. **URL & Email Stripping:** Removes hyper-links (`http://`, `https://`) and email addresses which do not carry generalizable sentiment.
3. **HTML Tag Removal:** Strips residual markup tags (`<p>`, `<div>`, etc.).
4. **Whitespace Normalization:** Collapses extraneous tabs, carriage returns, and multiple spaces.
5. **Sentiment Punctuation Preservation:** Unlike destructive preprocessing pipelines that strip all punctuation, SentimentScope deliberately preserves exclamation marks (`!`) and question marks (`?`) because they carry strong affective sentiment weight in subjective text.

---

## 4. Methodology & Theoretical Formulations

### 4.1 Classical Baseline: TF-IDF + Logistic Regression

#### Feature Representation (TF-IDF)
For a term $t$ in a document $d$ within a corpus of $N$ documents:
$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$
$$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

To capture local phrase structure and negations (e.g. "not bad", "never disappointing"), SentimentScope configures:
- N-gram Range: Unigrams and Bigrams `(1, 2)`.
- Max Features: 30,000 top vocabulary terms.
- Sublinear TF Scaling: $1 + \log(\text{TF})$ to dampen the impact of repeated words.
- Minimum Document Frequency: $\text{min\_df} = 2$.

#### Classification: Logistic Regression
The logit $z$ is computed as the inner product of learned weight vector $w$ and input vector $x$ plus bias $b$:
$$z = w^T x + b$$
The class probability $P(Y = 1 | x)$ is modeled via the standard logistic sigmoid:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
Objective optimization minimizes binary cross-entropy with L2 regularization penalty:
$$\mathcal{L}(w) = -\frac{1}{M}\sum_{i=1}^M \left[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right] + \frac{1}{2C}\|w\|_2^2$$

### 4.2 Deep Learning Paradigm: DistilBERT

DistilBERT (Sanh et al., 2019) is a distilled version of BERT (Bidirectional Encoder Representations from Transformers). It utilizes:
- **Multi-Head Self-Attention:** Computes pairwise token interactions across all positions simultaneously:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **Triple Loss Knowledge Distillation:** Combines hard label loss (cross-entropy), soft target distillation loss (Kullback-Leibler divergence with teacher model logits), and cosine embedding distance.
- **Architectural Efficiency:** Retains 97% of BERT-base's language comprehension performance with 40% fewer parameters and 60% faster inference latency.

---

## 5. Experimental Evaluation & Empirical Results

All evaluations were executed on the official SST-2 validation partition consisting of **872 labelled sentences** (444 positive, 428 negative).

### Comparative Metric Summary Table

| Metric | Baseline: TF-IDF + Logistic Regression | Transformer: DistilBERT (SST-2) |
| :--- | :--- | :--- |
| **Accuracy** | **80.73%** (0.8073) | **91.06%** (0.9106) |
| **Precision** | **0.7840** | **0.8978** |
| **Recall** | **0.8581** | **0.9302** |
| **F1-Score** | **0.8194** | **0.9137** |
| **ROC-AUC** | **0.9087** | **0.9716** |
| **Inference Latency** | **< 0.5 ms** (CPU) | **~15-25 ms** (CPU) |
| **Model Disk Size** | **~3.2 MB** | **~268 MB** |
| **Training Time** | **< 5 seconds** | Pre-trained / Fine-tuned |

*(Note: Evaluated strictly on the official SST-2 validation partition of 872 labelled sentences).*

---

## 6. Confusion Matrix Analysis

### Baseline Model Matrix (TF-IDF + Logistic Regression)
```
                  Predicted Negative (0)    Predicted Positive (1)
Actual Negative (0)        323 (TN)                  105 (FP)
Actual Positive (1)         63 (FN)                  381 (TP)
```
- **True Negatives (TN):** 323 correctly identified negative reviews.
- **False Positives (FP):** 105 negative reviews mistakenly predicted as positive (Type I error).
- **False Negatives (FN):** 63 positive reviews mistakenly predicted as negative (Type II error).
- **True Positives (TP):** 381 correctly identified positive reviews.

### Transformer Model Matrix (DistilBERT SST-2)
```
                  Predicted Negative (0)    Predicted Positive (1)
Actual Negative (0)        381 (TN)                   47 (FP)
Actual Positive (1)         31 (FN)                  413 (TP)
```
- **True Negatives (TN):** 381 correctly identified negative reviews (+58 over baseline).
- **False Positives (FP):** 47 false positives (-58 reduction in Type I error).
- **False Negatives (FN):** 31 false negatives (-32 reduction in Type II error).
- **True Positives (TP):** 413 correctly identified positive reviews (+32 over baseline).

---

## 7. Engineering Trade-off Analysis & Discussion

| Dimension | TF-IDF + Logistic Regression | DistilBERT (Transformer) |
| :--- | :--- | :--- |
| **Throughput / Latency** | Extreme throughput (> 5,000 queries/sec on 1 core) | Moderate throughput (~50-100 queries/sec on CPU) |
| **Hardware Requisites** | Minimal CPU, negligible memory footprint (< 100 MB RAM) | Substantial RAM (> 1 GB) and benefits from GPU/TPU |
| **Complex Syntax / Sarcasm**| Limited by linear combination of n-gram tokens | Superior context tracking across distant clauses |
| **Interpretability** | Transparent: feature weights reveal exact words | Opaque black box requiring attention visualization / SHAP |

---

## 8. Limitations & Future Directions

1. **Sarcasm and Subtlety:** Sarcastic sentences containing superficially positive vocabulary ("What a wonderful way to ruin a Saturday afternoon") remain challenging for n-gram models without deep semantic framing.
2. **Domain Adaptation:** Both models are evaluated on movie review syntax. Generalizing to financial disclosures or medical feedback may require domain-adaptive pre-training.
3. **Quantization & Edge Optimization:** Exploring ONNX runtime quantization (INT8) for DistilBERT could reduce memory consumption and boost CPU throughput by 2-3x.

---

## 9. Viva Voce & Interview Questions (Beginner-Friendly)

### Q1: What is data leakage, and how did you prevent it in your project?
**Answer:** Data leakage occurs when information from outside the training partition (such as validation or test data) contaminates the training phase, resulting in overly optimistic, invalid performance scores. In SentimentScope, we prevented data leakage by strictly fitting the `TfidfVectorizer` solely on the 67,349 training sentences. The validation set of 872 sentences was strictly transformed using the previously learned vocabulary and IDF weights, ensuring the model never saw test token distributions.

### Q2: Why did you use TF-IDF instead of simple word counts (Bag of Words)?
**Answer:** Simple word count gives high frequency to ubiquitous stop words like "the", "a", and "is", which carry no sentiment information. TF-IDF penalizes these common words across the corpus via Inverse Document Frequency, giving higher numerical weights to rare, discriminative words like "exquisite", "abysmal", "thrilling", or "disastrous".

### Q3: Why is DistilBERT preferred over original BERT for web applications?
**Answer:** Original BERT has 110 million parameters and can be sluggish on standard CPU servers. DistilBERT reduces the parameter count to 66 million via knowledge distillation, running 60% faster while maintaining 97% of BERT's performance, making it much more viable for interactive web deployment.

### Q4: Why is Accuracy alone not always sufficient to evaluate a classifier?
**Answer:** If a dataset has class imbalance (e.g. 95% positive, 5% negative), a trivial classifier that always predicts positive achieves 95% accuracy while being completely useless at detecting negatives. Precision, Recall, and the F1-Score (harmonic mean of Precision and Recall) provide a balanced evaluation across both classes.

---

## 10. Presentation Outline (5-10 Minute Walkthrough)

- **Slide 1: Title & Motivation:** SentimentScope: Scalable & Rigorous Sentiment Analysis.
- **Slide 2: Problem Statement & Objectives:** Balancing accuracy vs. computational efficiency in production.
- **Slide 3: Dataset (SST-2):** Stanford Sentiment Treebank structure, train/val split, and data leakage safeguards.
- **Slide 4: Model 1 - Classical Baseline:** TF-IDF unigram+bigram representation and Logistic Regression sigmoid mathematics.
- **Slide 5: Model 2 - Transformer Architecture:** DistilBERT self-attention and transfer learning.
- **Slide 6: Empirical Results & Confusion Matrices:** Honest comparative metrics on 872 validation sentences.
- **Slide 7: Live Web Demonstration:** Real-time Streamlit app walk-through with example test sentences.
- **Slide 8: Key Takeaways & Trade-offs:** When to choose lightweight linear models vs. heavy deep learning models.
