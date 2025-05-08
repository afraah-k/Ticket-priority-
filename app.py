import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
# Add local nltk_data to the data path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)


# Download required NLTK resources
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Define lemmatizer and preprocessing function
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(lemmatized)

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("best_support_ticket_model_compressed.pkl")
    vectorizer = joblib.load("tfidf_vectorizer_compressed.pkl")
    return model, vectorizer

# Initialize app
st.title("Support Ticket Priority Predictor")

try:
    model, vectorizer = load_model_and_vectorizer()
except Exception as e:
    st.error(f"Failed to load model/vectorizer: {e}")
    st.stop()

# Input text
user_input = st.text_area("Enter your support ticket text:")

if st.button("Predict Priority"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        try:
            # Preprocess input
            processed_input = [lemmatize_text(user_input)]
            vectorized_input = vectorizer.transform(processed_input)
            prediction = model.predict(vectorized_input)[0]
            st.success(f"Predicted Priority: **{prediction}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

            
