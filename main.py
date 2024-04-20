
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



# Download NLTK resources

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


data = pd.read_csv("data.csv")

data.head()

plt.bar(x = ['1', '0'], height = data['generated'].value_counts())


data.columns


# Function for text preprocessing
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




# Preprocess text data
data['processed_text'] = data['text'].apply(preprocess_text)

# %%
X = data['processed_text']
y = data['generated']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Convert sparse matrices to numpy arrays
X_train_vectors = X_train_vectors.toarray()
X_test_vectors = X_test_vectors.toarray()

# Calculate cosine similarity with reference vector
human_reference_vector = np.mean(X_train_vectors[y_train == 0], axis=0)  # Reference vector for human-written essays
cosine_similarity_train = cosine_similarity(X_train_vectors, [human_reference_vector])
cosine_similarity_test = cosine_similarity(X_test_vectors, [human_reference_vector])

# Train logistic regression model using cosine similarity scores
model = LogisticRegression()
model.fit(cosine_similarity_train, y_train)

# Predict labels for the testing set
y_pred = model.predict(cosine_similarity_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# %%
#export vectorizor and lr model
import pickle

# Save vectorizer
with open('lr_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Save logistic regression model
with open('lr_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# %%
del model

# %%
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Prepare features (X) and labels (y)
X = data['processed_text']
y = data['generated']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Convert sparse matrices to numpy arrays
X_train_vectors = X_train_vectors.toarray()
X_test_vectors = X_test_vectors.toarray()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Define neural network architecture
model = Sequential([
    Dense(64, input_dim=X_train_vectors.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
history = model.fit(X_train_vectors, y_train, epochs=100, batch_size=32, validation_data=(X_test_vectors, y_test), callbacks=[early_stopping])

# Evaluate model performance
loss, accuracy = model.evaluate(X_test_vectors, y_test)
print("Accuracy:", accuracy)

# %%
from tensorflow.keras.models import save_model

# Save the model
save_model(model, 'nn_model.keras')

# %%
from sklearn.metrics import confusion_matrix

# Predict probabilities for each class
y_pred_probabilities = model.predict(X_test_vectors)

# Convert probabilities to binary predictions
y_pred = (y_pred_probabilities > 0.5).astype(int)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

# %%



