"""Reusable feature building: TF-IDF + 6 hand-crafted features.
Used by models/train_model.py now, and by evaluation/ and app/ later."""
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

KEYWORDS = ["free", "win", "winner", "prize", "urgent", "claim", "cash", "txt", "reply"]


def hand_features(raw):
    raw = str(raw)
    low = raw.lower()
    n = max(len(raw), 1)
    return [
        len(re.findall(r"http\S+|www\.\S+", low)),   # number of links
        raw.count("!"),                               # exclamation marks
        sum(c.isupper() for c in raw) / n,            # share of capital letters
        sum(c.isdigit() for c in raw) / n,            # share of digits
        len(raw.split()),                             # word count
        sum(k in low for k in KEYWORDS),              # spam keywords present
    ]


def hand_matrix(raw_texts):
    return np.array([hand_features(x) for x in raw_texts], dtype=float)


def fit_hand_scale(raw_texts):
    """Max per column, computed on TRAIN only."""
    mx = hand_matrix(raw_texts).max(axis=0)
    mx[mx == 0] = 1
    return mx


def build_features(cleaned_texts, raw_texts, vectorizer, hand_max):
    X = vectorizer.transform(cleaned_texts)
    H = csr_matrix(hand_matrix(raw_texts) / hand_max)
    return hstack([X, H]).tocsr()