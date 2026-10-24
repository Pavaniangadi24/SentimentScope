"""TF-IDF + Logistic Regression Baseline Model for SST-2 Sentiment Analysis.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocessor import clean_text, batch_clean_texts
from src.utils import calculate_metrics, ID_TO_LABEL, LABEL_TO_ID

DEFAULT_MODEL_DIR = Path("models/baseline")


class BaselineSentimentClassifier:
    """TF-IDF and Logistic Regression pipeline for Sentiment Analysis."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 30000,
        sublinear_tf: bool = True,
        min_df: int = 2,
        c_param: float = 1.0,
        random_state: int = 42
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.c_param = c_param
        self.random_state = random_state

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.is_fitted: bool = False

    def fit(self, train_texts: List[str], train_labels: List[int]) -> "BaselineSentimentClassifier":
        """Fits the TF-IDF vectorizer and Logistic Regression classifier strictly on training data.
        
        To prevent data leakage:
        - The vectorizer vocabulary and IDF weights are computed solely from train_texts.
        - Labels are strictly binary (0: Negative, 1: Positive).
        """
        print(f"Preprocessing {len(train_texts):,} training texts...")
        cleaned_train = batch_clean_texts(train_texts)

        print("Fitting TF-IDF Vectorizer...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            strip_accents="unicode"
        )
        x_train = self.vectorizer.fit_transform(cleaned_train)
        y_train = np.array(train_labels)

        print(f"TF-IDF feature matrix shape: {x_train.shape}")
        print("Training Logistic Regression model...")
        self.classifier = LogisticRegression(
            C=self.c_param,
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs"
        )
        self.classifier.fit(x_train, y_train)
        self.is_fitted = True
        print("Model training complete.")
        return self

    def evaluate(self, val_texts: List[str], val_labels: List[int]) -> Dict[str, Any]:
        """Evaluates the model on validation data (no fitting to prevent data leakage)."""
        if not self.is_fitted or self.vectorizer is None or self.classifier is None:
            raise ValueError("Model has not been fitted yet. Call fit() or load() first.")

        print(f"Evaluating on {len(val_texts):,} validation samples...")
        cleaned_val = batch_clean_texts(val_texts)
        x_val = self.vectorizer.transform(cleaned_val)
        y_val = np.array(val_labels)

        y_pred = self.classifier.predict(x_val).tolist()
        y_prob = self.classifier.predict_proba(x_val)[:, 1].tolist()

        metrics = calculate_metrics(y_true=y_val.tolist(), y_pred=y_pred, y_prob=y_prob)
        metrics["model_name"] = "TF-IDF + Logistic Regression"
        metrics["evaluation_dataset"] = "SST-2 Validation Split (872 labelled sentences)"
        metrics["hyperparameters"] = {
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "min_df": self.min_df,
            "C": self.c_param,
            "random_state": self.random_state
        }
        self.metrics = metrics
        return metrics

    def predict(self, text: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Predicts sentiment label and class probabilities for one or multiple texts."""
        if not self.is_fitted or self.vectorizer is None or self.classifier is None:
            raise ValueError("Model has not been fitted or loaded.")

        is_single = isinstance(text, str)
        text_list = [text] if is_single else text

        cleaned_list = batch_clean_texts(text_list)
        x_input = self.vectorizer.transform(cleaned_list)
        preds = self.classifier.predict(x_input)
        probs = self.classifier.predict_proba(x_input)

        results = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            pred_id = int(pred)
            results.append({
                "label": ID_TO_LABEL[pred_id],
                "label_id": pred_id,
                "confidence": float(prob[pred_id]),
                "probabilities": {
                    "Negative": float(prob[0]),
                    "Positive": float(prob[1])
                },
                "cleaned_text": cleaned_list[i]
            })

        return results[0] if is_single else results

    def save(self, model_dir: Union[str, Path] = DEFAULT_MODEL_DIR) -> None:
        """Saves vectorizer, classifier, and evaluation metrics to disk."""
        if not self.is_fitted or self.vectorizer is None or self.classifier is None:
            raise ValueError("Cannot save an unfitted model.")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        vectorizer_path = model_dir / "tfidf_vectorizer.joblib"
        classifier_path = model_dir / "logistic_regression.joblib"
        metrics_path = model_dir / "baseline_metrics.json"

        print(f"Saving vectorizer to {vectorizer_path}...")
        joblib.dump(self.vectorizer, vectorizer_path)

        print(f"Saving classifier to {classifier_path}...")
        joblib.dump(self.classifier, classifier_path)

        if self.metrics:
            print(f"Saving baseline metrics to {metrics_path}...")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(self.metrics, f, indent=2)

        print("Baseline model artifacts successfully saved.")

    def load(self, model_dir: Union[str, Path] = DEFAULT_MODEL_DIR) -> "BaselineSentimentClassifier":
        """Loads fitted vectorizer, classifier, and saved metrics from disk."""
        model_dir = Path(model_dir)
        vectorizer_path = model_dir / "tfidf_vectorizer.joblib"
        classifier_path = model_dir / "logistic_regression.joblib"
        metrics_path = model_dir / "baseline_metrics.json"

        if not vectorizer_path.exists() or not classifier_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {model_dir}. Please train the model first."
            )

        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)
        self.is_fitted = True

        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        print(f"Baseline model successfully loaded from {model_dir}.")
        return self


def train_and_evaluate_baseline() -> BaselineSentimentClassifier:
    """End-to-end training and evaluation of the baseline model on SST-2."""
    from src.data_loader import load_sst2_data

    train_df, val_df = load_sst2_data()

    classifier = BaselineSentimentClassifier()
    classifier.fit(train_df["sentence"].tolist(), train_df["label"].tolist())
    metrics = classifier.evaluate(val_df["sentence"].tolist(), val_df["label"].tolist())

    print("\n" + "=" * 50)
    print("BASELINE EVALUATION RESULTS (SST-2 Validation Split)")
    print("=" * 50)
    print(f"Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    if metrics.get("roc_auc"):
        print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("-" * 50)
    print(f"Total Validation Samples: {metrics['total_samples']}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']['matrix']}")
    print("=" * 50)

    classifier.save()
    return classifier


if __name__ == "__main__":
    train_and_evaluate_baseline()
