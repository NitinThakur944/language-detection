"""Quick command-line smoke test for the saved model."""
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "language_model.pkl")
vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")

print("Supported model labels:", ", ".join(str(x) for x in model.classes_))
text = input("Enter a sentence to detect its language: ").strip()
if not text:
    raise SystemExit("No text entered.")
features = vectorizer.transform([text])
prediction = model.predict(features)[0]
print(f"Detected language: {prediction}")
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(features)[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True)[:5]
    print("Top predictions:")
    for language, probability in ranked:
        print(f"  {language}: {probability * 100:.2f}%")
