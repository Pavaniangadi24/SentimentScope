"""Unit tests for SentimentScope preprocessing, metrics, and baseline model.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.preprocessor import clean_text, batch_clean_texts
from src.utils import calculate_metrics, ID_TO_LABEL
from src.baseline_model import BaselineSentimentClassifier


def test_clean_text_removes_urls_and_html():
    raw = "Check this out: <p>Best movie ever! https://example.com/review &amp; must-watch</p>"
    cleaned = clean_text(raw)
    assert "https://" not in cleaned
    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert "Best movie ever!" in cleaned


def test_clean_text_handles_none_and_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   \n\t  ") == ""


def test_calculate_metrics():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 0, 0]  # 4 out of 5 correct, 1 false negative
    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.8
    assert metrics["total_samples"] == 5
    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"]["true_negatives"] == 2
    assert metrics["confusion_matrix"]["true_positives"] == 2
    assert metrics["confusion_matrix"]["false_negatives"] == 1
    assert metrics["confusion_matrix"]["false_positives"] == 0


def test_baseline_fit_predict_save_load():
    train_texts = [
        "A truly magnificent and delightful masterpiece.",
        "Loved the story and captivating acting.",
        "Terrible acting, awful plot, hated every second.",
        "Boring, sluggish, and a complete waste of time.",
        "Inspiring, heartwarming, and brilliant direction.",
        "Dull, lifeless, and poorly edited mess."
    ]
    train_labels = [1, 1, 0, 0, 1, 0]

    val_texts = [
        "Delightful and brilliant cinematic experience.",
        "Awful, boring, and terrible screenplay."
    ]
    val_labels = [1, 0]

    model = BaselineSentimentClassifier(min_df=1, max_features=100)
    model.fit(train_texts, train_labels)

    eval_res = model.evaluate(val_texts, val_labels)
    assert "accuracy" in eval_res
    assert eval_res["accuracy"] >= 0.5

    pred = model.predict("This movie was wonderful and delightful!")
    assert pred["label"] in ["Positive", "Negative"]
    assert "probabilities" in pred
    assert 0.0 <= pred["probabilities"]["Positive"] <= 1.0

    # Test saving and loading
    temp_dir = tempfile.mkdtemp()
    try:
        model.save(temp_dir)
        loaded = BaselineSentimentClassifier()
        loaded.load(temp_dir)
        pred_loaded = loaded.predict("This movie was wonderful and delightful!")
        assert pred_loaded["label"] == pred["label"]
    finally:
        shutil.rmtree(temp_dir)
