"""Data loading and caching module for SST-2 dataset.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from datasets import load_dataset


DEFAULT_CACHE_DIR = Path("data/processed")


def load_sst2_data(
    cache_dir: Optional[Path] = None,
    force_download: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the SST-2 dataset from Hugging Face or local parquet cache.
    
    The SST-2 (Stanford Sentiment Treebank v2) binary classification dataset:
      - Train split: 67,349 labelled sentences (0 = negative, 1 = positive)
      - Validation split: 872 labelled sentences (0 = negative, 1 = positive)
      - Test split: 1,821 unlabelled sentences (test set in SST-2 is unlabelled for benchmark evaluation)
      
    To prevent data leakage, training strictly occurs on the `train` split,
    and all reported baseline and transformer evaluation strictly occurs on the `validation` split.
    
    Args:
        cache_dir: Directory to cache processed parquet files.
        force_download: If True, re-downloads from Hugging Face.
        
    Returns:
        Tuple of (train_df, validation_df) with columns ['sentence', 'label'].
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
        
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    train_cache = cache_dir / "sst2_train.parquet"
    val_cache = cache_dir / "sst2_validation.parquet"
    
    if not force_download and train_cache.exists() and val_cache.exists():
        print(f"Loading cached SST-2 dataset from {cache_dir}...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        return train_df, val_df
        
    print("Downloading SST-2 dataset from Hugging Face (stanfordnlp/sst2)...")
    dataset = load_dataset("stanfordnlp/sst2")
    
    train_df = pd.DataFrame({
        "sentence": dataset["train"]["sentence"],
        "label": dataset["train"]["label"]
    })
    
    val_df = pd.DataFrame({
        "sentence": dataset["validation"]["sentence"],
        "label": dataset["validation"]["label"]
    })
    
    # Clean up empty rows or invalid entries if any
    train_df = train_df.dropna(subset=["sentence", "label"])
    val_df = val_df.dropna(subset=["sentence", "label"])
    
    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)
    
    print(f"Saving cached splits to {cache_dir}...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    
    print(f"SST-2 Loaded: {len(train_df):,} train samples, {len(val_df):,} validation samples.")
    return train_df, val_df


if __name__ == "__main__":
    train_df, val_df = load_sst2_data()
    print("Train sample:")
    print(train_df.head())
    print("\nValidation sample:")
    print(val_df.head())
