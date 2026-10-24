"""Text preprocessing and cleaning utilities for SentimentScope.
"""

import re
import html
from typing import List, Union


def clean_text(text: Union[str, float, None]) -> str:
    """Cleans input text for sentiment analysis.
    
    Operations performed:
    1. Handle None / non-string gracefully.
    2. Unescape HTML entities (&amp; -> &, etc.).
    3. Remove URLs and email addresses.
    4. Normalize excessive whitespace.
    5. Retain punctuation nuances (such as exclamation marks and question marks)
       which carry strong emotional/sentiment weight.
    
    Args:
        text: Raw text string or object.
        
    Returns:
        Cleaned text string.
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML characters
    cleaned = html.unescape(text)
    
    # 2. Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", "", cleaned)
    
    # 3. Remove email addresses
    cleaned = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "", cleaned)
    
    # 4. Remove HTML tags if present
    cleaned = re.sub(r"<.*?>", "", cleaned)
    
    # 5. Normalize whitespace (tabs, newlines, multiple spaces)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned


def batch_clean_texts(texts: List[str]) -> List[str]:
    """Clean a batch of text strings.
    
    Args:
        texts: List of raw input texts.
        
    Returns:
        List of cleaned texts.
    """
    return [clean_text(t) for t in texts]
