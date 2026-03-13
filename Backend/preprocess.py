import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure nltk resources are available
def initialize_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except (LookupError, AttributeError):
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except (LookupError, AttributeError):
        nltk.download('wordnet')

initialize_nltk()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove MBTI types from text to avoid data leakage/overfitting
    types = ['infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp', 
             'isfp', 'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj']
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Lowercase
    text = text.lower()
    # Remove the types
    for t in types:
        text = text.replace(t, '')
    # Tokenize and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

def preprocess_mbti_data(df):
    # Data Preparation: Handling imperfections (if any)
    # For MBTI, usually we just clean the text
    df['clean_posts'] = df['posts'].apply(clean_text)
    return df

