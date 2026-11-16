"""Train and compare spam classifiers on the VALIDATION set.
Run from the project root:  python -m models.train_model
The test set is NOT touched here (that is for evaluation/)."""
import joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score

from models.featurize import fit_hand_scale, build_features

train = pd.read_csv("data/train.csv").dropna()
val = pd.read_csv("data/val.csv").dropna()

# Vectorizer was fitted on TRAIN only by features/spam_features.py
vec = joblib.load("features/vectorizer.pkl")
hand_max = fit_hand_scale(train["raw_text"])

Xtr = build_features(train["text"], train["raw_text"], vec, hand_max)
Xva = build_features(val["text"], val["raw_text"], vec, hand_max)
ytr, yva = train["label"], val["label"]

neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
models = {
    "naive_bayes": MultinomialNB(alpha=0.1),
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "linear_svm": LinearSVC(class_weight="balanced"),
}
try:
    from xgboost import XGBClassifier
    models["xgboost"] = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=neg / pos, eval_metric="logloss", random_state=42)
except ImportError:
    print("xgboost not installed, skipping it")

rows, fitted = [], {}
for name, clf in models.items():
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)
    rows.append([name,
                 round(precision_score(yva, pred), 4),
                 round(recall_score(yva, pred), 4),
                 round(f1_score(yva, pred), 4)])
    fitted[name] = clf

results = pd.DataFrame(rows, columns=["model", "precision", "recall", "val_f1"])
results = results.sort_values("val_f1", ascending=False).reset_index(drop=True)
print(results.to_string(index=False))
results.to_csv("models/model_comparison.csv", index=False)

best = results.loc[0, "model"]
print(f"\nBest model on validation F1: {best}")

# One bundle with everything the app needs to predict on a new message
joblib.dump({"model_name": best, "model": fitted[best],
             "vectorizer": vec, "hand_max": hand_max},
            "models/best_model.pkl")
print("Saved models/best_model.pkl")