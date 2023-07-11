import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize NLTK components
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to transform text
def transform_text(text):
    # Lower case
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Removing stopwords and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)

# Load the saved models
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Set Streamlit page title and header
st.set_page_config(page_title='SMS Spam Classifier')
st.title('SMS Spam Classifier')

# Get user input message
message = st.text_input('Enter the message', placeholder=None)

# Preprocess the message
transformed_message = transform_text(message)

# Vectorize the transformed_text
vector_input = tfidf.transform([transformed_message])

# Perform prediction
result = model.predict(vector_input)[0]

# Display prediction upon button click
if st.button('Predict'):
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
