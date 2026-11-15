import re, joblib
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

train = pd.read_csv("data/train.csv").dropna()
val = pd.read_csv("data/val.csv").dropna()

# TF-IDF fitted on TRAIN only (no leakage)
vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=50000)
Xtr = vec.fit_transform(train["text"])
Xva = vec.transform(val["text"])
joblib.dump(vec, "features/vectorizer.pkl")

KEYWORDS = ["free", "win", "winner", "prize", "urgent", "claim", "cash", "txt", "reply"]
def hand(raw):
    raw = str(raw); low = raw.lower(); n = max(len(raw), 1)
    return [len(re.findall(r"http\S+|www\.\S+", low)), raw.count("!"),
            sum(c.isupper() for c in raw) / n, sum(c.isdigit() for c in raw) / n,
            len(raw.split()), sum(k in low for k in KEYWORDS)]

Htr = np.array([hand(x) for x in train["raw_text"]], dtype=float)
Hva = np.array([hand(x) for x in val["raw_text"]], dtype=float)
mx = Htr.max(axis=0); mx[mx == 0] = 1 # scale using train only
Htr, Hva = csr_matrix(Htr / mx), csr_matrix(Hva / mx)

rows = []
for name, (A, B) in {"tfidf_only": (Xtr, Xva),
                     "tfidf_plus_handcrafted": (hstack([Xtr, Htr]), hstack([Xva, Hva]))}.items():
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(A, train["label"])
    rows.append([name, round(f1_score(val["label"], clf.predict(B)), 4)])
out = pd.DataFrame(rows, columns=["features", "val_f1"])
out.to_csv("features/feature_comparison.csv", index=False)
print(out)