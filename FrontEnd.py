import streamlit as st
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#from main import preprocess_text

# Load the trained model and vectorizer
model = joblib.load('lr_model.pkl')
vectorizer = joblib.load('lr_vectorizer.pkl')


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    return processed_text

def main():
    st.title("AI Detection System")

    # Input text box for user to enter text
    user_input = st.text_area("Enter the text you want to analyze", "")
    if user_input != "":
        data = preprocess_text(user_input)
        
        print(data)



main()