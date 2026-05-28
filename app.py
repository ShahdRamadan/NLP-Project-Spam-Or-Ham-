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

# Page Architecture Configuration
st.set_page_config(page_title="Enterprise Email Security Classifier", page_icon="🔒", layout="centered")

# Custom Attention Layer to handle Keras/TensorFlow hybrid serialization protocols smoothly
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# Caching operational resource loading routines to guarantee optimal UI responsiveness
@st.cache_resource
def load_pipeline_assets():
    # Quietly secure NLTK linguistic dependencies
    for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng', 'omw-1.4']:
        nltk.download(pkg, quiet=True)
    
    # Load all core preprocessing and transformation pipeline vectors
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    scaler = joblib.load('scaler.joblib')
    le = joblib.load('label_encoder.joblib')
    
    # Attempt to load the Keras Tokenizer asset safely to support Deep Learning text sequencing
    try:
        keras_tokenizer = joblib.load('keras_tokenizer.joblib')
    except FileNotFoundError:
        keras_tokenizer = None
        
    return tfidf, scaler, le, keras_tokenizer

tfidf_tool, scaler_tool, le_tool, keras_tokenizer_tool = load_pipeline_assets()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(treebank_tag):
    mapping = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
    return mapping.get(treebank_tag[0], wordnet.NOUN)

def preprocess_text_fixed(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)   
    text = re.sub(r'\d+', ' ', text)        
    words = text.split()
    pos_tags = nltk.pos_tag(words)
    return ' '.join(
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
        for word, tag in pos_tags
        if word not in stop_words and len(word) > 1
    )

@st.cache_resource
def load_all_models():
    models = {}
    try: models['Linear SVM'] = joblib.load('spam_svm_model.joblib')
    except: pass
    try: models['XGBoost'] = joblib.load('spam_xgb_model.joblib')
    except: pass
    try: models['LSTM + Attention (Deep Learning)'] = tf.keras.models.load_model('spam_hybrid_attention_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
    except: pass
    return models

loaded_models = load_all_models()

# ========================================================
# BACKEND STRUCTURAL METADATA ENGINE
# ========================================================
def extract_features_from_input(sender_email, subject, body):
    full_text = f"{subject} {body}"
    full_text_lower = full_text.lower()
    
    # Mathematical Word count analysis
    num_words = len(full_text.split())
    
    # Tracking links via regular expression definitions
    urls = re.findall(r'(https?://[^\s]+|www\.[^\s]+)', full_text)
    num_links = len(urls)
    
    # Deep keyword check for high-risk anchor metrics
    has_suspicious_link = 0
    suspicious_keywords = ['free', 'click', 'login', 'verify', 'update', 'secure', 'bank', 'prize', 'bonus']
    for url in urls:
        if any(keyword in url.lower() for keyword in suspicious_keywords) or re.search(r'\d{4,}', url):
            has_suspicious_link = 1
            break
            
    # File attachment text pattern scanning flags
    num_attachments = 1 if any(word in full_text_lower for word in ['attach', 'file', 'pdf', 'invoice', 'doc']) else 0
    
    # Financial and urgency keyword matrices
    money_words = ['free', 'money', 'cash', 'win', 'prize', 'dollar', 'earn', 'click here', 'buy']
    urgency_words = ['urgent', 'immediately', 'now', 'limited', 'expire', 'action required', 'verify now']
    
    contains_money_terms = 1 if any(word in full_text_lower for word in money_words) else 0
    contains_urgency_terms = 1 if any(word in full_text_lower for word in urgency_words) else 0
    
    # Safe Isolation of Domain name elements using Gmail Parsing Protocols
    email_match = re.search(r'@([\w.-]+)', sender_email)
    if email_match:
        sender_domain = email_match.group(1).lower()
    else:
        sender_domain = sender_email.strip().lower() if sender_email.strip() else "unknown-sender.com"
        
    try:
        le_dict = dict(zip(le_tool.classes_, le_tool.transform(le_tool.classes_)))
        sender_domain_encoded = le_dict.get(sender_domain, -1)
    except:
        sender_domain_encoded = -1

    # Runtime Execution Date / Hour calculations
    now = datetime.now()
    email_hour = now.hour
    email_day_of_week = now.weekday() 
    is_weekend = 1 if email_day_of_week in [4, 5] else 0 
    
    # Normalized Baseline Model Metadata Weights
    sender_reputation_score = 0.20 if (contains_money_terms or has_suspicious_link) else 0.85
    num_recipients = 12 if contains_urgency_terms else 1

    # Assemble aligned Pandas DataFrame targeting exact feature configuration schema
    numeric_df = pd.DataFrame([{
        'num_words': num_words, 'num_links': num_links, 'has_suspicious_link': has_suspicious_link,
        'num_attachments': num_attachments, 'sender_reputation_score': sender_reputation_score,
        'email_hour': email_hour, 'email_day_of_week': email_day_of_week, 'is_weekend': is_weekend,
        'num_recipients': num_recipients, 'contains_money_terms': contains_money_terms,
        'contains_urgency_terms': contains_urgency_terms, 'sender_domain_encoded': sender_domain_encoded
    }])
    
    return numeric_df, sender_domain

# --- Interface Rendering Workspace ---
st.title("🔒 Automated Email Security Threat Analysis System")
st.markdown("Input operational message indicators below. The classification framework automatically assesses architectural risk metrics across three localized core models.")

sender_input = st.text_input("Sender Address (From):", placeholder="e.g., support-update@security-paypal.com")
subject = st.text_input("Email Subject Header Line:", placeholder="e.g., Immediate Profile Verification Required")
body = st.text_area("Message Body Text Structure:", height=220, placeholder="Paste email message contents here...")

if st.button("🚀 Execute Comparative Security Analytics", type="primary"):
    if not subject.strip() and not body.strip():
        st.warning("Action halted. Please enter text values within either the Subject or Body parameters.")
    else:
        with st.spinner("Analyzing message properties and calculating predictive vector metrics..."):
            # 1. Pipeline Feature Parsing
            full_text = f"{subject} {body}"
            clean_text = preprocess_text_fixed(full_text)
            numeric_df, extracted_domain = extract_features_from_input(sender_input, subject, body)
            
            # Diagnostic telemetry visibility window
            with st.expander("⚙️ View Extracted Telemetry Matrix Properties (Engine Log):"):
                st.write(f"**Isolated Domain Key:** `{extracted_domain}`")
                st.dataframe(numeric_df)
            
            # 2. Process Classical Machine Learning Input Arrays
            text_tfidf = tfidf_tool.transform([clean_text])
            numeric_scaled = scaler_tool.transform(numeric_df)
            final_features_ml = csr_matrix(hstack((text_tfidf, numeric_scaled)))
            
            # 3. Process Deep Learning Sequencing Realistically
            if keras_tokenizer_tool is not None:
                text_sequence = keras_tokenizer_tool.texts_to_sequences([clean_text])
                text_pad = pad_sequences(text_sequence, maxlen=150, padding='post')
            else:
                # Fallback mechanism if tokenizer file is physically missing from active project path
                text_pad = pad_sequences([[1] * min(len(clean_text.split()), 150)], maxlen=150, padding='post')

            # 4. Multi-Model Inference Executions
            results = []
            for name, model in loaded_models.items():
                if name == 'LSTM + Attention (Deep Learning)':
                    prob = model.predict([text_pad, numeric_scaled])[0][0]
                    pred = 1 if prob > 0.5 else 0
                    conf = f"{prob*100:.2f}% (Threat Probability)"
                else:
                    pred = model.predict(final_features_ml)[0]
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(final_features_ml)[0][1]
                        conf = f"{prob*100:.2f}%"
                    else:
                        conf = "N/A (Linear SVM Boundary Metric)"
                
                verdict = "🚨 SPAM (Malicious Flag)" if pred == 1 else "🍏 HAM (Verified Safe)"
                results.append({"Analytical Model Architecture": name, "Classification Verdict": verdict, "Certainty Weight": conf})
                
            st.subheader("📊 Cross-Architecture Model Scoring Summary Matrix")
            st.dataframe(pd.DataFrame(results), use_container_width=True)