import streamlit as st
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the trained model and vectorizer

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


st.set_page_config(layout="wide")
st.title("AI Detection System")
col1, col2 = st.columns([2,1])


l_main = ['Bag of Word', 'Cosine Similarity', "N-Gram"]
select_main = col2.selectbox('Select Feature Engineering', l_main, index=None)

#Cosine
if select_main == l_main[1]:
    vectorizer = joblib.load('cosine_vectorizer.pkl')
    l = ['Logistic Regression', 'Neural Network']
    select = col2.selectbox('Select Model', l, index=None)
    

    if select == l[0]:
        model = joblib.load('lr_model.pkl')
        desc = 'The logistic regression model is a simple linear model trained on TF-IDF vectors representing text data. It utilizes a linear combination of input features without any hidden layers. The model is trained using the logistic function to perform binary classification.'
        col2.write('Model Description')
        col2.write(desc)

    elif select == l[1]:
        model = load_model('nn_model.keras')
        desc = 'The neural network model comprises an input layer that accepts TF-IDF vectors representing text data, followed by two hidden layers with 64 and 32 neurons, respectively, utilizing ReLU activation functions. The output layer consists of a single neuron with a sigmoid activation function for binary classification. The model is trained using the Adam optimizer with binary cross-entropy loss. Early stopping with a patience of 3 epochs is employed to prevent overfitting.'
        col2.write('Model Description')
        col2.write(desc)

#BOW
elif select_main == l_main[0]:
    vectorizer = joblib.load('count_vectorizer.pkl')
    l = ['SVM', 'Logistic Regression']
    select = col2.selectbox('Select Model', l, index=None)
    

    if select == l[0]:
        model = joblib.load('svm_bow.pkl')
        desc = 'Support Vector Machine is a linear model that separates classes by finding the hyperplane that maximizes the margin between them. In this case, SVM is trained on bag-of-words features to classify text data into different categories using a linear kernel.'
        col2.write('Model Description')
        col2.write(desc)

    elif select == l[1]:
        model = joblib.load('lr_model_bow.pkl')
        desc = 'Logistic Regression is a simple linear model trained on bag-of-words features extracted from text data. It calculates the probability of a sample belonging to a particular class using a logistic function and performs binary classification.'
        col2.write('Model Description')
        col2.write(desc)

elif select_main == l_main[2]:
    vectorizer = joblib.load('vectorizer_ngram.pkl')
    l = ['SVM', 'Logistic Regression']
    select = col2.selectbox('Select Model', l, index=None)
    

    if select == l[0]:
        model = joblib.load('svc_ngram.pkl')
        desc = 'Our Support Vector Machine (SVM) model, utilizing the n-grams approach, effectively classifies text data into distinct categories. By analyzing sequences of words and phrases (n-grams), SVM constructs a hyperplane that maximizes the margin between different text classes. This approach ensures robust text classification, making our system adept at distinguishing between various types of text.'
        col2.write('Model Description')
        col2.write(desc)

    elif select == l[1]:
        model = joblib.load('lr_ngram.pkl')
        desc = 'Utilizing the n-grams approach, our Logistic Regression model efficiently analyzes text data to classify it into distinct categories. By considering both single words (unigrams) and pairs of consecutive words (bigrams), Logistic Regression learns patterns in text data and makes accurate predictions. This approach provides a straightforward yet effective method for text classification, ensuring the reliability of our system.'
        col2.write('Model Description')
        col2.write(desc)

# Input text box for user to enter text
user_input = col1.text_area("Enter the essay you want to analyze", "")



pred = col1.button('Predict')
if pred :
    # Preprocess the user input
    processed_input = preprocess_text(user_input)

    # Vectorize the preprocessed input
    input_vector = vectorizer.transform([processed_input])

    
    # Predict using the model
    prediction = model.predict(input_vector[0])
    
    # Display the prediction
    if prediction > 0.5:
        col1.write("The text is generated by AI.")
        st.image('the-rock-rock.gif')
    else:
        col1.write("The text is human-written.")


st.markdown(
    """
    <footer style="padding:10px; text-align:left;">
        <p>About Project</p>
        <p>Our AI text detection system employs three distinct approaches: Bag-of-Words (BoW), N-grams, and Cosine Similarity. Within each approach, we have trained two models, each optimized for text classification tasks. For BoW and N-grams, Logistic Regression and Support Vector Machine (SVM) models are utilized, while for Cosine Similarity, K-Nearest Neighbors (KNN) and Naive Bayes classifiers are employed. Evaluation of these models is based on accuracy, measuring the proportion of correctly classified texts. Higher accuracy indicates better performance in distinguishing between different types of text data, ensuring the effectiveness of our text detection system.</p>
    </footer>
    """,
    unsafe_allow_html=True
)