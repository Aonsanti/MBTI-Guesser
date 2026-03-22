import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
import os

# Ensure nltk resources are available
def initialize_nltk():
    import tempfile
    # Vercel serverless has a read-only filesystem except for /tmp
    nltk_data_dir = os.path.join(tempfile.gettempdir(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
        
    try:
        nltk.data.find('corpora/stopwords')
    except (LookupError, AttributeError):
        nltk.download('stopwords', download_dir=nltk_data_dir)
    try:
        nltk.data.find('corpora/wordnet')
    except (LookupError, AttributeError):
        nltk.download('wordnet', download_dir=nltk_data_dir)

initialize_nltk()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# ==================== IMDB Preprocessing (for ML model) ====================

def clean_text_imdb(text):
    """
    Clean and preprocess text for IMDB sentiment analysis.
    Handles HTML tags (via BeautifulSoup), URLs, special characters, and applies lemmatization.
    """
    # Remove HTML tags robustly
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except Exception:
        # Fallback to regex if BS4 fails
        text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers, keep only letters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Lowercase
    text = text.lower()
    
    # Sentiment-critical words we MUST keep
    keep_words = {'not', 'no', 'never', 'nor', 'neither', 'none', 'cannot', 'without', 'against'}
    
    # Tokenize and lemmatize
    words = text.split()
    cleaned_words = []
    for word in words:
        if word in keep_words:
            cleaned_words.append(word)
        elif word not in stop_words and len(word) > 2:
            cleaned_words.append(lemmatizer.lemmatize(word))
            
    return " ".join(cleaned_words)

def preprocess_imdb_data(df):
    """
    Data Preparation for IMDB Dataset:
    - Handle missing values
    - Remove duplicate reviews
    - Clean HTML tags, URLs, special chars
    - Apply lemmatization and stopword removal
    - Encode sentiment labels (positive=1, negative=0)
    """
    df = df.dropna(subset=['review', 'sentiment'])
    df = df.drop_duplicates(subset=['review'])
    
    # Parallelize the cleaning process
    print(f"Cleaning {len(df)} reviews in parallel...")
    num_cores = os.cpu_count() or 1
    df['clean_review'] = Parallel(n_jobs=num_cores)(
        delayed(clean_text_imdb)(review) for review in df['review']
    )
    
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df


# ==================== MBTI Preprocessing (for NN model) ====================

def clean_text_mbti(text):
    """
    Clean and preprocess text for MBTI personality prediction.
    Removes URLs, MBTI type mentions, special characters, applies lemmatization.
    """
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
    """
    Data Preparation for MBTI Dataset:
    - Clean text posts
    """
    df['clean_posts'] = df['posts'].apply(clean_text_mbti)
    return df

