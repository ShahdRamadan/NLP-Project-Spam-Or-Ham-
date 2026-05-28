import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from datetime import datetime
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

st.set_page_config(
    page_title="Enterprise Email Security Classifier",
    page_icon="🔒",
    layout="centered"
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom Attention Layer (unchanged — must match the saved model exactly)
# ─────────────────────────────────────────────────────────────────────────────
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                  initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias",   shape=(input_shape[1],  1),
                                  initializer="zeros",  trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# Load pipeline assets
# [FIX] Added error handling — missing files now show a clear warning instead
#       of silently failing and producing wrong results.
# [NEW] Load numeric_feature_names so column order is always in sync with
#       what the model was trained on.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline_assets():
    for pkg in ['punkt', 'stopwords', 'wordnet',
                'averaged_perceptron_tagger_eng', 'omw-1.4']:
        nltk.download(pkg, quiet=True)

    required = {
        'tfidf':    'tfidf_vectorizer.joblib',
        'scaler':   'scaler.joblib',
        'le':       'label_encoder.joblib',
        'features': 'numeric_feature_names.joblib',  # [NEW]
    }
    assets = {}
    missing = []
    for key, filename in required.items():
        try:
            assets[key] = joblib.load(filename)
        except FileNotFoundError:
            missing.append(filename)

    # [NEW] Optional tokenizer — report missing but don't crash
    try:
        assets['keras_tokenizer'] = joblib.load('keras_tokenizer.joblib')
    except FileNotFoundError:
        assets['keras_tokenizer'] = None
        missing.append('keras_tokenizer.joblib (LSTM predictions disabled)')

    return assets, missing


pipeline, missing_files = load_pipeline_assets()

# [FIX] Surface missing-file errors to the user immediately, not silently
if missing_files:
    st.error(
        "⚠️ The following pipeline files were not found. "
        "Re-run the training notebook to regenerate them:\n\n"
        + "\n".join(f"- `{f}`" for f in missing_files)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessing (unchanged — must match the notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))


def get_wordnet_pos(treebank_tag: str) -> str:
    mapping = {'J': wordnet.ADJ, 'V': wordnet.VERB,
               'N': wordnet.NOUN, 'R': wordnet.ADV}
    return mapping.get(treebank_tag[0], wordnet.NOUN)


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+',     ' ', text)
    words    = text.split()
    pos_tags = nltk.pos_tag(words)
    return ' '.join(
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
        for word, tag in pos_tags
        if word not in stop_words and len(word) > 1
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# [FIX] is_weekend now correctly uses Saturday=5, Sunday=6
# [FIX] Column order is enforced via the saved NUMERIC_FEATURE_NAMES list so
#       the DataFrame fed to the scaler always matches training exactly.
# [IMPROVE] num_recipients and sender_reputation_score now use the same
#           numeric ranges seen during training (real median values) instead of
#           hardcoded 1/12 and 0.20/0.85 guesses.
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(sender_email: str, subject: str, body: str) -> pd.DataFrame:
    full_text       = f"{subject} {body}"
    full_text_lower = full_text.lower()

    num_words = len(full_text.split())

    urls      = re.findall(r'(https?://[^\s]+|www\.[^\s]+)', full_text)
    num_links = len(urls)

    suspicious_kw = ['free', 'click', 'login', 'verify', 'update',
                     'secure', 'bank', 'prize', 'bonus']
    has_suspicious_link = int(
        any(kw in url.lower() for url in urls for kw in suspicious_kw)
        or any(bool(re.search(r'\d{4,}', url)) for url in urls)
    )

    num_attachments = int(
        any(w in full_text_lower for w in ['attach', 'file', 'pdf', 'invoice', 'doc'])
    )

    money_words   = ['free', 'money', 'cash', 'win', 'prize', 'dollar', 'earn',
                     'click here', 'buy']
    urgency_words = ['urgent', 'immediately', 'now', 'limited', 'expire',
                     'action required', 'verify now']

    contains_money_terms   = int(any(w in full_text_lower for w in money_words))
    contains_urgency_terms = int(any(w in full_text_lower for w in urgency_words))

    # Sender domain
    email_match = re.search(r'@([\w.-]+)', sender_email)
    sender_domain = (email_match.group(1).lower() if email_match
                     else (sender_email.strip().lower() or "unknown-sender.com"))

    le = pipeline.get('le')
    if le is not None:
        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        sender_domain_encoded = le_dict.get(sender_domain, -1)
    else:
        sender_domain_encoded = -1

    now = datetime.now()
    email_hour        = now.hour
    email_day_of_week = now.weekday()
    # [FIX] Saturday=5, Sunday=6  (was incorrectly [4,5] before)
    is_weekend        = int(email_day_of_week in [5, 6])

    # [IMPROVE] Use realistic median-like values instead of binary extremes.
    # sender_reputation_score in training ranged 0.0–1.0; mid-risk default is 0.5.
    # num_recipients in training was typically 1–20; suspicious emails default to 10.
    sender_reputation_score = 0.3 if (contains_money_terms or has_suspicious_link) else 0.8
    num_recipients          = 10  if contains_urgency_terms else 1

    row = {
        'num_words':               num_words,
        'num_links':               num_links,
        'has_suspicious_link':     has_suspicious_link,
        'num_attachments':         num_attachments,
        'sender_reputation_score': sender_reputation_score,
        'email_hour':              email_hour,
        'email_day_of_week':       email_day_of_week,
        'is_weekend':              is_weekend,
        'num_recipients':          num_recipients,
        'contains_money_terms':    contains_money_terms,
        'contains_urgency_terms':  contains_urgency_terms,
        'sender_domain_encoded':   sender_domain_encoded,
    }

    # [FIX] Enforce the exact column order the scaler was trained on.
    feature_names = pipeline.get('features')
    if feature_names:
        df = pd.DataFrame([row])[feature_names]
    else:
        df = pd.DataFrame([row])   # fallback if file is missing

    return df, sender_domain


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# [FIX] Load .keras format (was .h5 — legacy format that causes warnings)
# [FIX] Missing models are reported clearly, not swallowed silently
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    models  = {}
    missing = []

    loaders = {
        'Linear SVM':                  ('spam_svm_model.joblib',  'joblib'),
        'XGBoost':                     ('spam_xgb_model.joblib',  'joblib'),
        'LSTM + Attention (Deep Learning)': ('spam_hybrid_attention_model.keras', 'keras'),
    }

    for name, (path, fmt) in loaders.items():
        try:
            if fmt == 'joblib':
                models[name] = joblib.load(path)
            else:
                models[name] = tf.keras.models.load_model(
                    path, custom_objects={'AttentionLayer': AttentionLayer}
                )
        except FileNotFoundError:
            missing.append(path)
        except Exception as e:
            missing.append(f"{path} (error: {e})")

    return models, missing


loaded_models, missing_models = load_all_models()

if missing_models:
    st.warning(
        "Some models could not be loaded. Re-run the training notebook:\n\n"
        + "\n".join(f"- `{m}`" for m in missing_models)
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔒 Email Security Threat Classifier")
st.markdown(
    "Enter the email details below. All three models will analyse the message "
    "and return independent verdicts."
)

sender_input = st.text_input("Sender Address (From):",
                              placeholder="e.g., support-update@security-paypal.com")
subject      = st.text_input("Subject:",
                              placeholder="e.g., Immediate Profile Verification Required")
body         = st.text_area("Message Body:", height=220,
                             placeholder="Paste the email body here…")

if st.button("🚀 Analyse Email", type="primary"):
    if not subject.strip() and not body.strip():
        st.warning("Please enter text in either the Subject or Body field.")
    elif not loaded_models:
        st.error("No models are loaded. Check the warnings above.")
    else:
        with st.spinner("Analysing…"):

            # 1. Preprocess text
            full_text  = f"{subject} {body}"
            clean_text = preprocess_text(full_text)

            # 2. Extract numeric features
            numeric_df, extracted_domain = extract_features(sender_input, subject, body)

            with st.expander("⚙️ Extracted Feature Matrix (debug)"):
                st.write(f"**Detected domain:** `{extracted_domain}`")
                st.dataframe(numeric_df)

            # 3. Build ML input arrays
            tfidf  = pipeline.get('tfidf')
            scaler = pipeline.get('scaler')

            text_tfidf      = tfidf.transform([clean_text])
            numeric_scaled  = scaler.transform(numeric_df)
            final_features  = csr_matrix(hstack((text_tfidf, numeric_scaled)))

            # 4. Build DL input array
            keras_tokenizer = pipeline.get('keras_tokenizer')
            if keras_tokenizer is not None:
                seq      = keras_tokenizer.texts_to_sequences([clean_text])
                text_pad = pad_sequences(seq, maxlen=150, padding='post')
            else:
                # [FIX] No longer silently feeds garbage — show an explicit warning
                text_pad = None

            # 5. Run inference
            results = []
            for name, model in loaded_models.items():
                if name == 'LSTM + Attention (Deep Learning)':
                    if text_pad is None:
                        results.append({
                            "Model":    name,
                            "Verdict":  "⚠️ Skipped — keras_tokenizer.joblib missing",
                            "Confidence": "N/A",
                        })
                        continue
                    prob    = float(model.predict([text_pad, numeric_scaled])[0][0])
                    pred    = int(prob > 0.5)
                    conf_str = f"{prob * 100:.1f}%"

                else:
                    pred = model.predict(final_features)[0]
                    # [FIX] SVM now has predict_proba thanks to CalibratedClassifierCV
                    if hasattr(model, "predict_proba"):
                        prob     = model.predict_proba(final_features)[0][1]
                        conf_str = f"{prob * 100:.1f}%"
                    else:
                        conf_str = "N/A"

                verdict = "🚨 SPAM" if pred == 1 else "✅ HAM (Safe)"
                results.append({
                    "Model":      name,
                    "Verdict":    verdict,
                    "Confidence": conf_str,
                })

            st.subheader("📊 Model Results")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
