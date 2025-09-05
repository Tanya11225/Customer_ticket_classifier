import pandas as pd
import re
import nltk

# Download required NLTK data (only once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords

# Get English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses a single text string
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

def load_and_clean_data(filepath):
    """
    Loads the CSV data and applies preprocessing
    """
    df = pd.read_csv(filepath)
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    return df