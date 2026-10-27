"""Train/evaluate the language detector from the included dataset."""
from pathlib import Path
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
DATA = BASE / "LanguageDetection.csv"
LABEL_FIX = {"Portugeese": "Portuguese", "Sweedish": "Swedish"}

def clean_text(value):
    # Keep Unicode letters from every language, rather than stripping non-Latin scripts.
    text = str(value).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def main():
    df = pd.read_csv(DATA)
    if not {"Text", "Language"}.issubset(df.columns):
        raise ValueError("Dataset needs Text and Language columns")
    df = df.dropna(subset=["Text", "Language"]).copy()
    df["Language"] = df["Language"].replace(LABEL_FIX)
    df["Text"] = df["Text"].map(clean_text)
    df = df[df["Text"].str.len() >= 2].drop_duplicates(subset=["Text", "Language"])
    x_train, x_test, y_train, y_test = train_test_split(
        df["Text"], df["Language"], test_size=0.20, random_state=42, stratify=df["Language"]
    )
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=30000, sublinear_tf=True, min_df=2)
    xtr = vectorizer.fit_transform(x_train)
    xte = vectorizer.transform(x_test)
    model = MultinomialNB(alpha=0.15)
    model.fit(xtr, y_train)
    pred = model.predict(xte)
    acc = accuracy_score(y_test, pred)
    print(f"Languages: {len(model.classes_)} | Samples: {len(df):,} | Held-out accuracy: {acc:.2%}")
    print(classification_report(y_test, pred, zero_division=0))
    cm = confusion_matrix(y_test, pred, labels=model.classes_)
    plt.figure(figsize=(13, 10))
    sns.heatmap(cm, cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"Language Detection — held-out accuracy {acc:.2%}")
    plt.xlabel("Predicted language"); plt.ylabel("Actual language")
    plt.xticks(rotation=60, ha="right"); plt.tight_layout()
    plt.savefig(BASE / "confusion_matrix.png", dpi=150); plt.close()
    joblib.dump(model, BASE / "language_model.pkl")
    joblib.dump(vectorizer, BASE / "vectorizer.pkl")
    print("Saved language_model.pkl and vectorizer.pkl")
if __name__ == "__main__": main()
