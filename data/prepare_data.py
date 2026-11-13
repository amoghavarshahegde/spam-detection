import pandas as pd

FILE = "data/mail_data.csv"

df = pd.read_csv(FILE, encoding="latin-1")
df = df.rename(columns={"Message": "text", "Category": "label"})[["text", "label"]]

df["label"] = df["label"].astype(str).str.strip().str.lower().map({"spam": 1, "ham": 0})

df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip().str.len() > 5]
df = df.drop_duplicates(subset="text").reset_index(drop=True)

df.to_csv("data/emails_raw.csv", index=False)
df.sample(n=100, random_state=42).to_csv("data/sample_100.csv", index=False)

print("Total messages:", len(df))
print(df["label"].value_counts())