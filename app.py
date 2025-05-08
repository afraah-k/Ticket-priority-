import streamlit as st
import joblib
import re

# --- Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Load model and vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("best_support_ticket_model_compressed.pkl")
    vectorizer = joblib.load("tfidf_vectorizer_compressed.pkl")
    return model, vectorizer

# --- Priority Mapping ---
priority_map = {
    "High": "P1",
    "Medium": "P2",
    "Low": "P3"
}

# --- Streamlit UI ---
st.title("Support Ticket Priority Predictor")

try:
    model, vectorizer = load_model_and_vectorizer()
except Exception as e:
    st.error(f"❌ Failed to load model/vectorizer: {e}")
    st.stop()

user_input = st.text_area("Enter your support ticket text:")

if st.button("Predict Priority"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        try:
            processed_input = [preprocess_text(user_input)]
            vectorized_input = vectorizer.transform(processed_input)
            raw_prediction = model.predict(vectorized_input)[0]
            mapped_priority = priority_map.get(raw_prediction, "Unknown")
            st.success(f"✅ Predicted Priority: **{mapped_priority}** ({raw_prediction})")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
