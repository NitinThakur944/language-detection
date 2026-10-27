# Language Detection Studio (NLP)

A Streamlit application for language identification using the project's existing **character TF-IDF + Multinomial Naive Bayes** model.

## Features

- Modern responsive Streamlit interface
- Single-text language detection and top-five probability estimates
- Batch detection by pasted lines or uploaded TXT, CSV, and JSON files
- Download batch results as CSV
- Optional translation using `deep-translator` (requires internet access)
- Session activity history with CSV export and clear-history option
- Model details and limitations for coursework demonstration
- Existing model/vectorizer retained

## Supported languages

The supplied model has 17 output classes from the included dataset. Translation choices are limited to these six languages. The dataset may contain other labels, but the shipped classifier does not predict them. Do not claim broader language support unless you train and evaluate a new model.

## Windows 11 + VS Code setup (Python 3.10 recommended)

1. Extract this ZIP to a short path, for example `C:\LanguageDetectionProjectNLP_Updated`.
2. Open VS Code, choose **File → Open Folder**, and select that folder.
3. Open **Terminal → New Terminal**. Confirm the terminal path is the project folder.
4. Create a virtual environment:

   ```powershell
   py -3.10 -m venv .venv
   ```

   If `py -3.10` is unavailable, install Python 3.10 first or use `python -m venv .venv`.
5. Activate it in PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.
6. Upgrade pip and install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
7. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```
8. Streamlit will show a local URL, normally `http://localhost:8501`. Open it in your browser. Keep the terminal open while using the app; press `Ctrl+C` to stop it.

## Quick model test (optional)

In the activated VS Code terminal:

```powershell
python test_model.py
```

## Retrain the baseline (optional)

The current `.pkl` files are already included, so retraining is not required to run the app. To retrain from `LanguageDetection.csv`:

```powershell
python language_detection.py
```

This replaces the model/vectorizer files and confusion-matrix image. Back up the current `.pkl` files first if you want to preserve the originals. The training script reports held-out accuracy; use that measured result rather than assuming an accuracy target.

## Troubleshooting

- **Model files missing:** make sure `language_model.pkl` and `vectorizer.pkl` are in the same folder as `app.py`.
- **Module not found:** activate `.venv` and run `python -m pip install -r requirements.txt` again.
- **Port 8501 is busy:** run `python -m streamlit run app.py --server.port 8502`.
- **Translation fails:** detection works offline; translation needs an internet connection and an available translation service.
- **Model compatibility warning:** the saved model was created with scikit-learn 1.6.1. The requirements pin that version to reduce compatibility risk.

## Project files

- `app.py` — Streamlit UI and features
- `language_model.pkl` — existing classifier
- `vectorizer.pkl` — existing fitted TF-IDF vectorizer
- `LanguageDetection.csv` — original dataset
- `language_detection.py` — training/evaluation script
- `test_model.py` — command-line smoke test
- `confusion_matrix.png` — supplied evaluation visualization
- `requirements.txt` — runtime dependencies
- `upgrade_roadmap.md` — original upgrade roadmap

## Academic note

The app preserves the supplied model. A probability score is not a guarantee of correctness, especially for short text, names, code-switching, and text outside the training distribution. Improving accuracy or adding more languages requires suitable training data and a fresh evaluation.

## Accuracy notes

Short greetings such as “hola” and “ciao” contain very little context and can be ambiguous for statistical models. Common greetings and distinct scripts have explicit handling. Prefer a full sentence for best results. Translation uses an online service and requires internet access.


### Translation troubleshooting

The Translate page tries Google Translate first. If the free Google endpoint rate-limits requests, the app automatically tries MyMemory. Translation needs an internet connection; if both providers are unavailable, wait and retry later. Automatic source detection for the fallback uses the local language detection model, so full sentences are recommended.
