import streamlit as st
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
import joblib
from main import preprocess_text

# Load the trained model and vectorizer
model = joblib.load('lr_model.pkl')
vectorizer = joblib.load('lr_vectorizer.pkl')




def main():
    st.title("AI Detection System")

    # Input text box for user to enter text
    user_input = st.text_area("Enter the text you want to analyze", "")
    data = preprocess_text(preprocess_text)
    print(data)



main()