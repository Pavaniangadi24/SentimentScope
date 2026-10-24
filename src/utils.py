"""Utility functions for evaluation metrics, plotting, and constants.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

ID_TO_LABEL = {0: "Negative", 1: "Positive"}
LABEL_TO_ID = {"Negative": 0, "Positive": 1}

SAMPLE_SENTENCES = [
    {
        "text": "A brilliantly crafted film with breathtaking visuals and deeply moving performances.",
        "expected": "Positive"
    },
    {
        "text": "The pacing was painfully sluggish, and the plot was entirely predictable.",
        "expected": "Negative"
    },
    {
        "text": "An unforgettable masterpiece that stays with you long after the credits roll.",
        "expected": "Positive"
    },
    {
        "text": "Terrible dialogue, wooden acting, and a complete waste of time.",
        "expected": "Negative"
    },
    {
        "text": "It has its flaws and cliches, but overall it is charming, warm, and thoroughly enjoyable.",
        "expected": "Positive"
    },
    {
        "text": "Despite having a talented cast, the movie collapses under awful writing and shoddy direction.",
        "expected": "Negative"
    }
]


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted positive class probabilities (optional).
        
    Returns:
        Dictionary of performance metrics.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    
    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    
    report_dict = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)
    
    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "total_samples": len(y_true_arr),
        "confusion_matrix": {
            "matrix": cm,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "labels": ["Negative (0)", "Positive (1)"]
        },
        "classification_report": report_dict
    }
    
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true_arr, y_prob))
            metrics["roc_auc"] = round(auc, 4)
        except Exception:
            metrics["roc_auc"] = None
            
    return metrics
