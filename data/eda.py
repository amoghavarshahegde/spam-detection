import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

df = pd.read_csv("data/emails_raw.csv")
names = {0: "Not spam", 1: "Spam"}

# Chart 1: spam vs not spam
counts = df["label"].value_counts().sort_index()
plt.figure(figsize=(5, 4))
plt.bar([names[i] for i in counts.index], counts.values, color=["steelblue", "tomato"])
plt.title("Spam vs Not Spam")
plt.ylabel("Number of messages")
plt.savefig("data/chart_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# Chart 2: message length
df["length"] = df["text"].str.split().str.len()
plt.figure(figsize=(6, 4))
for lab in [0, 1]:
    plt.hist(df[df["label"] == lab]["length"].clip(upper=100), bins=40, alpha=0.6, label=names[lab])
plt.legend()
plt.title("Message length (words)")
plt.xlabel("Words")
plt.savefig("data/chart_length.png", dpi=150, bbox_inches="tight")
plt.close()

# Chart 3: top words
stop = set("the a an and or to of in is it for on you your i we this that with be are as at by from me my so but not have can will".split())
def top_words(series, n=15):
    words = re.findall(r"[a-z]{3,}", " ".join(series).lower())
    return Counter(w for w in words if w not in stop).most_common(n)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, lab in zip(axes, [0, 1]):
    w, c = zip(*top_words(df[df["label"] == lab]["text"]))
    ax.barh(w[::-1], c[::-1])
    ax.set_title(f"Top words: {names[lab]}")
plt.tight_layout()
plt.savefig("data/chart_top_words.png", dpi=150)

print("Total:", len(df))
print("Spam %:", round(df["label"].mean() * 100, 1))
print("Avg words, not spam:", round(df[df["label"] == 0]["length"].mean(), 1))
print("Avg words, spam:", round(df[df["label"] == 1]["length"].mean(), 1))