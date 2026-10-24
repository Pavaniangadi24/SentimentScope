"""Transformer (DistilBERT) model implementation and evaluation for SST-2 Sentiment Analysis.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

import numpy as np

DEFAULT_TRANSFORMER_DIR = Path("models/transformer")
DEFAULT_MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


class TransformerSentimentClassifier:
    """DistilBERT Transformer sentiment classifier."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None
    ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = device
        self.metrics: Optional[Dict[str, Any]] = None
        self.is_loaded: bool = False

    def load(self) -> "TransformerSentimentClassifier":
        """Loads the tokenizer and model from Hugging Face or local path."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Loading Transformer model '{self.model_name_or_path}' on device '{self.device}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print("Transformer model loaded successfully.")
            return self
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            raise e

    def predict(self, text: Union[str, List[str]], batch_size: int = 32) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Runs sentiment prediction using DistilBERT."""
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            self.load()

        import torch

        is_single = isinstance(text, str)
        text_list = [text] if is_single else text

        results = []
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

            for b_idx in range(len(batch_texts)):
                pred_id = int(preds[b_idx])
                prob_neg = float(probs[b_idx][0])
                prob_pos = float(probs[b_idx][1])
                # SST-2 DistilBERT: index 0 is NEGATIVE, index 1 is POSITIVE
                label = "Positive" if pred_id == 1 else "Negative"
                confidence = prob_pos if pred_id == 1 else prob_neg

                results.append({
                    "label": label,
                    "label_id": pred_id,
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        "Negative": round(prob_neg, 4),
                        "Positive": round(prob_pos, 4)
                    },
                    "cleaned_text": batch_texts[b_idx]
                })

        return results[0] if is_single else results

    def evaluate(
        self,
        val_texts: List[str],
        val_labels: List[int],
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Evaluates DistilBERT on the exact same validation examples used for the baseline."""
        from src.utils import calculate_metrics

        if not self.is_loaded:
            self.load()

        print(f"Evaluating DistilBERT on {len(val_texts):,} validation samples...")
        predictions = self.predict(val_texts, batch_size=batch_size)

        y_true = val_labels
        y_pred = [p["label_id"] for p in predictions]
        y_prob = [p["probabilities"]["Positive"] for p in predictions]

        metrics = calculate_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
        metrics["model_name"] = "DistilBERT (Fine-tuned on SST-2)"
        metrics["base_checkpoint"] = self.model_name_or_path
        metrics["device"] = self.device
        metrics["evaluation_dataset"] = "SST-2 Validation Split (872 labelled sentences)"
        self.metrics = metrics
        return metrics

    def save_metrics(self, output_dir: Union[str, Path] = DEFAULT_TRANSFORMER_DIR) -> None:
        """Saves evaluation metrics to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = output_dir / "transformer_metrics.json"

        if self.metrics:
            print(f"Saving transformer metrics to {metrics_file}...")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(self.metrics, f, indent=2)


def evaluate_transformer() -> Optional[Dict[str, Any]]:
    """Evaluates DistilBERT on the SST-2 validation dataset."""
    from src.data_loader import load_sst2_data

    _, val_df = load_sst2_data()

    clf = TransformerSentimentClassifier()
    try:
        metrics = clf.evaluate(val_df["sentence"].tolist(), val_df["label"].tolist())
        clf.save_metrics()

        print("\n" + "=" * 50)
        print("TRANSFORMER EVALUATION RESULTS (SST-2 Validation Split)")
        print("=" * 50)
        print(f"Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1-Score  : {metrics['f1']:.4f}")
        if metrics.get("roc_auc"):
            print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
        print("-" * 50)
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']['matrix']}")
        print("=" * 50)
        return metrics
    except Exception as e:
        print(f"Transformer evaluation failed or was skipped: {e}")
        return None


if __name__ == "__main__":
    evaluate_transformer()
