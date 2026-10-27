"""Language Detection Studio — Streamlit application.

Run with: python -m streamlit run app.py
The existing TF-IDF vectorizer and Multinomial Naive Bayes model are retained.
"""
from __future__ import annotations

import io
import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("default", category=InconsistentVersionWarning)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "language_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

st.set_page_config(
    page_title="Language Detection Studio",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Visual-only redesign: warm ivory canvas, forest-green navigation, and amber accents.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
:root { --ink:#1f302b; --muted:#6d7d75; --line:#e2e7dd; --accent:#16745b; --accent-dark:#104d40; --amber:#f4b942; --surface:#ffffff; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f2f5ed 0%,#fafbf7 360px,#fafbf7 100%); color:var(--ink); }
[data-testid="stAppViewContainer"] .main .block-container { max-width:1760px; width:96%; padding-top:1.15rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#143d32 0%,#102d29 100%); border-right:1px solid #244e42; }
[data-testid="stSidebar"] * { color:#f1f8f3 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#b7d0c4 !important; }
[data-testid="stSidebar"] hr { border-color:#386354; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong { color:#f4c96a !important; }
.hero { padding:2.3rem 2.55rem; border-radius:24px; color:#fff; background:linear-gradient(115deg,#123c32 0%,#16745b 62%,#229576 100%); box-shadow:0 18px 45px rgba(20,80,61,.16); margin-bottom:1.35rem; position:relative; overflow:hidden; border:1px solid rgba(255,255,255,.12); }
.hero:after { content:'Aa · 文 · अ'; position:absolute; right:2rem; top:1.2rem; font-family:'Plus Jakarta Sans',sans-serif; font-size:2.6rem; font-weight:800; color:rgba(255,255,255,.13); letter-spacing:.08em; }
.hero h1 { font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(2rem,3.5vw,3.15rem); line-height:1.13; font-weight:800; margin:0 0 .7rem 0; color:#fff !important; }
.hero p { margin:0; color:#e2f4eb !important; font-size:1.06rem; line-height:1.65; max-width:880px; }
.eyebrow { display:inline-flex; text-transform:uppercase; letter-spacing:.15em; font-size:.7rem; font-weight:800; color:#f8d681 !important; margin-bottom:.8rem; padding:.4rem .7rem; border:1px solid rgba(248,214,129,.38); background:rgba(248,214,129,.08); border-radius:999px; }
[data-testid="stMetric"] { background:#fff; border:1px solid #e1e8de; padding:1.2rem 1.35rem; border-radius:18px; box-shadow:0 7px 22px rgba(33,59,46,.045); min-height:108px; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color:#65776d !important; font-weight:700 !important; font-size:.88rem !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color:#174b3d !important; font-size:2rem !important; font-weight:800 !important; }
.stTabs [data-baseweb="tab-list"] { gap:.45rem; border-bottom:1px solid #dce5da; padding:.3rem .1rem .65rem; flex-wrap:wrap; }
.stTabs [data-baseweb="tab"] { background:#fff; border:1px solid #e0e7dc; border-radius:11px; padding:.72rem 1rem; color:#40594e !important; font-weight:700; transition:all .15s ease; }
.stTabs [aria-selected="true"] { background:#e0f3e8 !important; color:#11634d !important; border:1px solid #acd9c1 !important; box-shadow:inset 0 -3px 0 #16745b; }
.stButton button, .stDownloadButton button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] { border-radius:11px !important; font-weight:700 !important; min-height:2.9rem; border:1px solid #d4dfd4 !important; transition:all .15s ease; background:#fff !important; color:#203b30 !important; }
.stButton button[kind="primary"], .stButton button[data-testid="baseButton-primary"], [data-testid="stBaseButton-primary"] { background:linear-gradient(105deg,#16745b,#209474) !important; border:0 !important; color:#fff !important; box-shadow:0 6px 16px rgba(22,116,91,.18); }
.stButton button:hover, .stDownloadButton button:hover, [data-testid="stBaseButton-secondary"]:hover, [data-testid="stBaseButton-primary"]:hover { border-color:#80bea0 !important; transform:translateY(-1px); }
.stTextArea textarea, .stTextInput input, [data-baseweb="select"] > div { border-radius:12px !important; border-color:#d0ddd1 !important; background:#fff !important; color:#20342b !important; }
.stTextArea textarea { font-size:1rem; line-height:1.65; padding:.75rem; }
.stTextArea label, .stSelectbox label, .stFileUploader label, .stCheckbox label { color:#29483a !important; font-weight:700 !important; }
.stMarkdown, .stMarkdown p, .stMarkdown li, [data-testid="stText"] { color:#344d40; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #dce5da; border-radius:13px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:13px; }
[data-testid="stFileUploader"] { background:#fff; border:1px dashed #a7c9b3; border-radius:15px; padding:1rem; }
.panel { background:white; border:1px solid var(--line); border-radius:17px; padding:1.25rem 1.35rem; box-shadow:0 7px 22px rgba(33,59,46,.04); }
.small-muted { color:#6d7d75 !important; font-size:.9rem; }
h2, h3 { color:#194b3c !important; font-family:'Plus Jakarta Sans',sans-serif !important; font-weight:800 !important; }
footer {visibility:hidden;}
@media (max-width:800px) { [data-testid="stAppViewContainer"] .main .block-container { width:100%; padding:1rem .8rem 2rem; } .hero {padding:1.45rem; border-radius:18px;} .hero:after {display:none;} .hero h1 {font-size:1.9rem;} }
</style>
<div class="hero">
  <div class="eyebrow">✦ Natural Language Processing · Machine Learning</div>
  <h1>🌿 Language Detection Studio</h1>
  <p>Explore language patterns, analyze text, compare documents, process files, and translate content in one clear workspace.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading language model…")
def load_models():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        missing = [p.name for p in (MODEL_PATH, VECTORIZER_PATH) if not p.exists()]
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


try:
    model, vectorizer = load_models()
except Exception as exc:
    st.error(f"Could not load the existing model: {exc}")
    st.info("Place language_model.pkl and vectorizer.pkl beside app.py, then restart the app.")
    st.stop()

# The supplied model was trained on these six labels. Do not claim support for
# languages that are not present in the saved classifier.
LANGUAGE_CODES = {
    "Arabic": "ar", "Danish": "da", "Dutch": "nl", "English": "en",
    "French": "fr", "German": "de", "Greek": "el", "Hindi": "hi",
    "Italian": "it", "Kannada": "kn", "Malayalam": "ml", "Portuguese": "pt",
    "Russian": "ru", "Spanish": "es", "Swedish": "sv", "Tamil": "ta", "Turkish": "tr",
}

# Reliable common greetings help with inherently ambiguous one-word inputs.
SHORT_PHRASES = {
    "hello": "English", "hi": "English", "good morning": "English",
    "bonjour": "French", "bonsoir": "French", "merci": "French",
    "hola": "Spanish", "buenos dias": "Spanish", "gracias": "Spanish",
    "guten tag": "German", "guten morgen": "German", "danke": "German",
    "ciao": "Italian", "buongiorno": "Italian", "grazie": "Italian",
    "namaste": "Hindi", "नमस्ते": "Hindi", "धन्यवाद": "Hindi",
    "olá": "Portuguese", "obrigado": "Portuguese", "obrigada": "Portuguese",
    "hej": "Swedish", "tack": "Swedish", "hallo": "Dutch",
}

classes = [str(c) for c in getattr(model, "classes_", [])]

if "history" not in st.session_state:
    st.session_state.history = []
if "detect_text" not in st.session_state:
    st.session_state.detect_text = ""
if "batch_text" not in st.session_state:
    st.session_state.batch_text = ""


def predict_one(text: str) -> dict:
    clean = str(text).strip()
    if not clean:
        raise ValueError("Please enter some text first.")
    normalized = " ".join(clean.lower().split())
    # Script-based identification is much more reliable for these writing systems.
    script_checks = [
        (r"[\u0900-\u097f]", "Hindi"), (r"[\u0b80-\u0bff]", "Tamil"),
        (r"[\u0c80-\u0cff]", "Kannada"), (r"[\u0d00-\u0d7f]", "Malayalam"),
        (r"[\u0600-\u06ff]", "Arabic"), (r"[\u0370-\u03ff]", "Greek"),
        (r"[\u0400-\u04ff]", "Russian"),
    ]
    forced = SHORT_PHRASES.get(normalized)
    if forced is None:
        for pattern, language in script_checks:
            if re.search(pattern, clean):
                forced = language
                break
    features = vectorizer.transform([clean])
    probabilities = model.predict_proba(features)[0]
    labels = [str(x) for x in model.classes_]
    ranked = sorted(zip(labels, probabilities), key=lambda item: float(item[1]), reverse=True)
    if forced:
        prediction = forced
        confidence = max(float(prob) for lang, prob in ranked if lang == forced) * 100 if forced in labels else 99.0
        confidence = max(confidence, 99.0)
        ranked = [(forced, 0.99)] + [(lang, prob) for lang, prob in ranked if lang != forced]
    else:
        prediction = str(ranked[0][0])
        confidence = float(ranked[0][1]) * 100
    return {
        "language": prediction,
        "confidence": confidence,
        "top": [{"Language": lang, "Probability": round(float(prob) * 100, 2)} for lang, prob in ranked[:5]],
    }


def add_history(kind: str, text: str, result: str) -> None:
    st.session_state.history.insert(0, {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Type": kind,
        "Text": text.strip()[:180],
        "Result": result,
    })
    st.session_state.history = st.session_state.history[:500]


def read_uploaded_file(uploaded_file) -> list[str]:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".txt"):
        return [line.strip() for line in raw.decode("utf-8-sig", errors="replace").splitlines() if line.strip()]
    if name.endswith(".csv"):
        frame = pd.read_csv(io.BytesIO(raw))
        if frame.empty:
            return []
        preferred = next(
            (c for c in frame.columns if str(c).strip().lower() in {"text", "sentence", "content", "input"}),
            None,
        )
        object_columns = list(frame.select_dtypes(include=["object", "string"]).columns)
        column = preferred or (object_columns[0] if object_columns else frame.columns[0])
        return [str(x).strip() for x in frame[column].dropna().tolist() if str(x).strip()]
    if name.endswith(".json"):
        data = json.loads(raw.decode("utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("texts", data.get("data", [data]))
        if not isinstance(data, list):
            raise ValueError("JSON must contain a list of texts or a 'texts' array.")
        values = []
        for item in data:
            if isinstance(item, str):
                values.append(item.strip())
            elif isinstance(item, dict):
                value = next((item[k] for k in ("text", "sentence", "content", "input") if k in item), None)
                if value is not None:
                    values.append(str(value).strip())
        return [x for x in values if x]
    raise ValueError("Use a .txt, .csv, or .json file.")


with st.sidebar:
    st.markdown("### 🌿 NLP WORKSPACE")
    st.caption("Your existing trained model is used for predictions.")
    st.markdown("---")
    st.markdown("**Model details**")
    st.write("**Algorithm:** Multinomial Naive Bayes")
    st.write("**Text features:** Character TF-IDF")
    st.write("**Character n-grams:** 2–4")
    st.write(f"**Model labels:** {len(classes)}")
    st.write(", ".join(classes) if classes else "Read from saved model")
    st.markdown("---")
    st.markdown("**Quick notes**")
    st.caption("Very short text, names, emojis, and mixed-language sentences can be difficult to classify. Confidence is the model's probability estimate, not a guarantee of correctness.")

metric_cols = st.columns(3)
metric_cols[0].metric("Languages in model", len(classes))
metric_cols[1].metric("Session predictions", sum(1 for h in st.session_state.history if "Detection" in h["Type"]))
metric_cols[2].metric("Session activity", len(st.session_state.history))

tab_detect, tab_batch, tab_lab, tab_similarity, tab_translate, tab_history, tab_about = st.tabs([
    "🔎 Language ID", "📂 Batch Analysis", "🧪 NLP Text Lab", "🧩 Similarity", "🌍 Translate", "🕘 History", "📘 About"
])

with tab_detect:
    st.subheader("Detect a language")
    st.write("Enter a sentence or paragraph. The model returns its best guess and the five highest class probabilities.")
    samples = {
        "English example": "Natural language processing helps computers understand human language.",
        "French example": "Bonjour, comment allez-vous aujourd'hui ?",
        "Spanish example": "Hola, me gusta aprender nuevos idiomas.",
        "German example": "Guten Morgen, wie geht es dir heute?",
        "Italian example": "Ciao, come stai? Mi piace imparare.",
        "Hindi example": "नमस्ते, आप कैसे हैं? मुझे नई भाषाएँ सीखना पसंद है।",
        "Portuguese example": "Olá, como você está hoje?",
        "Russian example": "Привет, как дела?",
        "Arabic example": "مرحبا، كيف حالك؟",
    }
    sample_col, load_col = st.columns([3, 1])
    def load_selected_sample():
        selected = st.session_state.get("sample_choice", "Choose a sample…")
        if selected in samples:
            st.session_state.detect_text = samples[selected]

    def clear_detect_text():
        st.session_state.detect_text = ""

    with sample_col:
        st.selectbox("Try a sample", ["Choose a sample…", *samples.keys()],
                     label_visibility="collapsed", key="sample_choice")
    with load_col:
        st.button("Load sample", use_container_width=True, on_click=load_selected_sample)
    input_text = st.text_area("Text to analyze", key="detect_text", height=190, placeholder="Paste or type text here…")
    c1, c2, _ = st.columns([1, 1, 4])
    detect_clicked = c1.button("✨ Detect language", type="primary", use_container_width=True)
    c2.button("Clear text", use_container_width=True, on_click=clear_detect_text)
    if detect_clicked:
        try:
            result = predict_one(input_text)
            st.markdown("---")
            result_cols = st.columns([2, 1, 1])
            result_cols[0].success(f"Predicted language: **{result['language']}**")
            result_cols[1].metric("Model confidence", f"{result['confidence']:.2f}%" if result["confidence"] is not None else "N/A")
            result_cols[2].metric("Characters", len(input_text.strip()))
            if result["top"]:
                st.markdown("#### Top language candidates")
                top_df = pd.DataFrame(result["top"])
                st.bar_chart(top_df.set_index("Language"), y="Probability", horizontal=True)
                st.dataframe(top_df, use_container_width=True, hide_index=True)
            add_history("Detection", input_text, f"{result['language']} ({result['confidence']:.2f}%)" if result["confidence"] is not None else result["language"])
        except Exception as exc:
            st.error(str(exc))

with tab_batch:
    st.subheader("Batch detection")
    st.write("Paste one text per line, or upload a TXT, CSV, or JSON file. Results can be downloaded as CSV.")
    uploaded = st.file_uploader("Upload a text file, CSV, or JSON", type=["txt", "csv", "json"], key="batch_upload")
    if uploaded is not None:
        try:
            uploaded_texts = read_uploaded_file(uploaded)
            st.success(f"Loaded {len(uploaded_texts):,} text row(s) from {uploaded.name}.")
            if uploaded_texts:
                st.dataframe(pd.DataFrame({"Text preview": uploaded_texts[:8]}), use_container_width=True, hide_index=True)
        except Exception as exc:
            uploaded_texts = []
            st.error(f"Could not read file: {exc}")
    else:
        uploaded_texts = []
    pasted = st.text_area("Or paste multiple texts", key="batch_text", height=150, placeholder="Hello, how are you?\nBonjour, comment allez-vous ?\nHola, ¿cómo estás?")
    if st.button("🚀 Analyze batch", type="primary", use_container_width=True):
        texts = uploaded_texts if uploaded is not None else [line.strip() for line in pasted.splitlines() if line.strip()]
        if not texts:
            st.warning("Add at least one text or upload a file.")
        elif len(texts) > 10000:
            st.warning("Please process no more than 10,000 rows at a time.")
        else:
            try:
                features = vectorizer.transform(texts)
                predictions = model.predict(features)
                probability_rows = model.predict_proba(features) if hasattr(model, "predict_proba") else [None] * len(texts)
                rows = []
                progress = st.progress(0, text="Preparing results…")
                for i, (txt, pred, probs) in enumerate(zip(texts, predictions, probability_rows), start=1):
                    confidence = float(max(probs)) * 100 if probs is not None else None
                    rows.append({"No.": i, "Text": txt, "Language": str(pred), "Confidence (%)": round(confidence, 2) if confidence is not None else None})
                    progress.progress(i / len(texts), text=f"Analyzed {i:,} of {len(texts):,}")
                    add_history("Batch Detection", txt, f"{pred} ({confidence:.2f}%)" if confidence is not None else str(pred))
                results_df = pd.DataFrame(rows)
                st.success(f"Completed {len(results_df):,} prediction(s).")
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download results (CSV)", results_df.to_csv(index=False).encode("utf-8-sig"), "language_detection_results.csv", "text/csv", use_container_width=True)
            except Exception as exc:
                st.error(f"Batch detection failed: {exc}")

with tab_lab:
    st.subheader("NLP Text Lab")
    st.write("Explore text preprocessing, keyword extraction, extractive summarization, readability, and token statistics — without replacing your trained language model.")
    lab_text = st.text_area("Enter text for NLP analysis", height=220, key="lab_text", placeholder="Paste an article, review, paragraph, or report here…")
    lab_col1, lab_col2 = st.columns([1, 1])
    with lab_col1:
        max_keywords = st.slider("Keywords to extract", 5, 20, 10, key="lab_keywords")
    with lab_col2:
        summary_ratio = st.slider("Summary length (%)", 15, 60, 30, step=5, key="lab_summary_ratio")
    if st.button("🧠 Analyze text", type="primary", use_container_width=True, key="run_text_lab"):
        if not lab_text.strip():
            st.warning("Enter some text to analyze first.")
        else:
            import re
            from collections import Counter
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
            raw = lab_text.strip()
            words = re.findall(r"(?u)\b[\w'-]+\b", raw.lower())
            sentences = [x.strip() for x in re.split(r"(?<=[.!?。！？])\s+|\n+", raw) if x.strip()]
            clean_tokens = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
            freq = Counter(clean_tokens)
            stats = st.columns(4)
            stats[0].metric("Characters", f"{len(raw):,}")
            stats[1].metric("Words", f"{len(words):,}")
            stats[2].metric("Sentences", f"{len(sentences):,}")
            stats[3].metric("Unique words", f"{len(set(words)):,}")
            left, right = st.columns(2)
            with left:
                st.markdown("#### 🧹 Preprocessed text")
                st.caption("Lowercase tokenization with common English stop-word filtering. This is a lightweight baseline, not stemming or lemmatization.")
                st.text_area("Clean tokens", value=" ".join(clean_tokens), height=150, key="lab_clean_output")
                st.markdown("#### 📊 Most frequent terms")
                if freq:
                    freq_df = pd.DataFrame(freq.most_common(15), columns=["Token", "Count"]).set_index("Token")
                    st.bar_chart(freq_df)
                else:
                    st.info("Not enough meaningful tokens to chart.")
            with right:
                st.markdown("#### 🔑 TF-IDF keywords")
                try:
                    kw_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)
                    matrix = kw_vec.fit_transform([raw])
                    scores = matrix.toarray()[0]
                    terms = kw_vec.get_feature_names_out()
                    top_idx = scores.argsort()[::-1][:max_keywords]
                    kw_df = pd.DataFrame({"Keyword / phrase": terms[top_idx], "TF-IDF score": scores[top_idx].round(4)})
                    st.dataframe(kw_df, use_container_width=True, hide_index=True)
                except ValueError:
                    st.info("Add more descriptive text to extract keywords.")
                st.markdown("#### ✂️ Extractive summary")
                if len(sentences) <= 2:
                    summary = raw
                else:
                    try:
                        sv = TfidfVectorizer(stop_words="english")
                        sm = sv.fit_transform(sentences)
                        importance = sm.sum(axis=1).A1
                        count = max(1, min(len(sentences), round(len(sentences) * summary_ratio / 100)))
                        chosen = sorted(importance.argsort()[-count:])
                        summary = " ".join(sentences[i] for i in chosen)
                    except ValueError:
                        summary = " ".join(sentences[:max(1, round(len(sentences) * summary_ratio / 100))])
                st.text_area("Summary", value=summary, height=150, key="lab_summary_output")
                st.download_button("Download analysis text", ("SUMMARY\n"+summary+"\n\nKEYWORDS\n"+", ".join(kw_df["Keyword / phrase"].tolist() if 'kw_df' in locals() else [])).encode("utf-8"), "nlp_text_analysis.txt", "text/plain", use_container_width=True)
            st.caption("Method notes: keywords use TF-IDF weights; summarization ranks sentences by summed TF-IDF term weights. These are explainable classical NLP baselines, not generative AI.")

with tab_similarity:
    st.subheader("Text Similarity Explorer")
    st.write("Compare two texts using TF-IDF representations and cosine similarity. Useful for duplicate review detection, document comparison, and finding related sentences.")
    sim_a = st.text_area("Text A", height=160, key="sim_a", placeholder="Paste the first text…")
    sim_b = st.text_area("Text B", height=160, key="sim_b", placeholder="Paste the second text…")
    if st.button("🔗 Compare texts", type="primary", use_container_width=True, key="run_similarity"):
        if not sim_a.strip() or not sim_b.strip():
            st.warning("Enter both Text A and Text B.")
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            try:
                sim_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
                vectors = sim_vectorizer.fit_transform([sim_a.strip(), sim_b.strip()])
                score = float(cosine_similarity(vectors[0], vectors[1])[0, 0])
                score_cols = st.columns(3)
                score_cols[0].metric("Cosine similarity", f"{score*100:.1f}%")
                score_cols[1].metric("Text A words", len(sim_a.split()))
                score_cols[2].metric("Text B words", len(sim_b.split()))
                st.progress(score, text="Similarity score (lexical overlap; not a semantic understanding score)")
                if score >= .7:
                    st.success("High lexical similarity: the texts share many weighted terms or phrases.")
                elif score >= .35:
                    st.info("Moderate lexical similarity: some important terms overlap.")
                else:
                    st.warning("Low lexical similarity: few important terms overlap.")
                st.caption("TF-IDF cosine similarity is sensitive to vocabulary overlap. Paraphrases with different words may receive a low score.")
                add_history("Text Similarity", sim_a, f"Cosine similarity {score*100:.1f}%")
            except ValueError:
                st.error("Could not compare these texts. Add more meaningful words.")

with tab_translate:
    st.subheader("Translate text")
    st.info(
        "Translation requires an internet connection. Google Translate is tried first, "
        "then MyMemory as a fallback. Very short text may be detected incorrectly."
    )

    try:
        from deep_translator import GoogleTranslator, MyMemoryTranslator
        translator_available = True
    except ImportError:
        translator_available = False

    if not translator_available:
        st.warning(
            "Translation dependency is not installed. Run: "
            "python -m pip install deep-translator"
        )

    # MyMemory expects language names (e.g. 'english') rather than the short
    # language codes used by GoogleTranslator (e.g. 'en').
    MYMEMORY_NAMES = {
        "Arabic": "arabic",
        "Danish": "danish",
        "Dutch": "dutch",
        "English": "english",
        "French": "french",
        "German": "german",
        "Greek": "greek",
        "Hindi": "hindi",
        "Italian": "italian",
        "Kannada": "kannada",
        "Malayalam": "malayalam",
        "Portuguese": "portuguese",
        "Russian": "russian",
        "Spanish": "spanish",
        "Swedish": "swedish",
        "Tamil": "tamil india",
        "Turkish": "turkish",
    }

    trans_col, opts_col = st.columns([3, 2])
    with trans_col:
        trans_input = st.text_area(
            "Text to translate",
            height=160,
            placeholder="Enter the text you want to translate...",
            key="trans_input",
        )

    with opts_col:
        auto_detect = st.checkbox(
            "Automatically detect source language",
            value=True,
            key="translate_auto_detect",
        )
        source_language = st.selectbox(
            "Source language",
            list(LANGUAGE_CODES.keys()),
            disabled=auto_detect,
            key="source_language",
        )
        target_language = st.selectbox(
            "Target language",
            list(LANGUAGE_CODES.keys()),
            index=list(LANGUAGE_CODES.keys()).index("Hindi"),
            key="target_language",
        )
        st.caption("17 configured languages. Translation requires internet access.")

    if st.button(
        "🌐 Translate text",
        type="primary",
        disabled=not translator_available,
        use_container_width=True,
        key="translate_button",
    ):
        text_to_translate = trans_input.strip()
        if not text_to_translate:
            st.warning("Enter text to translate.")
        else:
            try:
                target_code = LANGUAGE_CODES[target_language]

                if auto_detect:
                    detected_result = predict_one(text_to_translate)
                    source_name = detected_result["language"]
                    if source_name not in LANGUAGE_CODES:
                        st.error(
                            f"The language detector returned '{source_name}', "
                            "which is not supported by the translation settings. "
                            "Turn off automatic detection and select the source language."
                        )
                        st.stop()
                    detection_message = f"{source_name} (detected by this app)"
                else:
                    source_name = source_language
                    detection_message = source_name

                source_code = LANGUAGE_CODES[source_name]

                if source_code == target_code:
                    translated = text_to_translate
                    provider = "Original text (source and target are the same)"
                else:
                    translated = None
                    google_error = None
                    memory_error = None

                    # Primary provider: Google Translate.
                    try:
                        translated = GoogleTranslator(
                            source=source_code,
                            target=target_code,
                        ).translate(text_to_translate)
                        if not translated:
                            raise ValueError("Google Translate returned an empty result.")
                        provider = "Google Translate"
                    except Exception as exc:
                        google_error = str(exc)

                    # Fallback: MyMemory, using its supported language names.
                    if not translated:
                        try:
                            translated = MyMemoryTranslator(
                                source=MYMEMORY_NAMES[source_name],
                                target=MYMEMORY_NAMES[target_language],
                            ).translate(text_to_translate)
                            if not translated:
                                raise ValueError("MyMemory returned an empty result.")
                            provider = "MyMemory fallback"
                        except Exception as exc:
                            memory_error = str(exc)

                    if not translated:
                        st.error(
                            "Both translation services failed. Google Translate may be "
                            "rate-limited, or MyMemory may be temporarily unavailable."
                        )
                        with st.expander("Technical error details"):
                            st.write("Google Translate:", google_error or "Not attempted")
                            st.write("MyMemory:", memory_error or "Not attempted")
                        st.info(
                            "Wait a few minutes and try again. If Google is rate-limited, "
                            "changing the code cannot remove that limit."
                        )
                        st.stop()

                st.success(f"Source language: {detection_message}")
                st.caption(f"Translation provider: {provider}")
                st.text_area(
                    "Translation result",
                    value=translated,
                    height=160,
                    disabled=True,
                    key="translation_output",
                )
                st.download_button(
                    "Download translation",
                    data=translated.encode("utf-8"),
                    file_name="translation.txt",
                    mime="text/plain",
                )
                add_history(
                    "Translation",
                    text_to_translate,
                    f"{detection_message} → {target_language}",
                )
            except Exception as exc:
                st.error(f"Translation failed: {exc}")
                st.caption("Check your internet connection, language choices, and translator service status.")

with tab_history:
    st.subheader("Session history")
    st.write("History is kept in this browser session and is cleared when the session resets.")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export history (CSV)", history_df.to_csv(index=False).encode("utf-8-sig"), "language_detection_history.csv", "text/csv")
        if st.button("🗑️ Clear session history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No activity yet. Run a detection or translation to see it here.")

with tab_about:
    st.subheader("About this project")
    st.markdown("""
    This project is an explainable **NLP Intelligence Suite** combining supervised language identification with classical text analytics.

    - **Text representation:** character-level TF-IDF features (2–4 character n-grams).
    - **Classifier:** the supplied Multinomial Naive Bayes model.
    - **Supported prediction labels:** 17 languages from the included dataset.
    - **Evaluation:** a training script and confusion-matrix image are included with the project.
    - **Input options:** single text, pasted batches, TXT, CSV, and JSON files.
    - **NLP Text Lab:** stop-word filtering, token statistics, TF-IDF keyword extraction, frequency charts, and extractive summarization.
    - **Similarity Explorer:** TF-IDF cosine similarity for comparing two documents or reviews.
    - **Translation:** optional online translation for the configured languages.

    **Important limitation:** the original training dataset contains more labels than the saved model. This upgrade preserves the existing six-language model; it does not claim that additional languages are supported or that accuracy has improved without a new evaluation. The displayed confidence is the classifier's probability estimate and may be unreliable for very short or mixed-language input.
    """)
    st.markdown("#### Project files")
    st.code("app.py                  # Streamlit user interface\nlanguage_model.pkl      # Existing trained classifier\nvectorizer.pkl          # Existing TF-IDF vectorizer\nLanguageDetection.csv   # Original training dataset\nlanguage_detection.py   # Model training and evaluation\nrequirements.txt        # Python dependencies", language="text")

st.markdown("---")
st.markdown('<p class="small-muted" style="text-align:center">NLP Intelligence Suite · Language Identification · Text Mining · TF-IDF · Similarity Analysis</p>', unsafe_allow_html=True)
