import re
import nltk
import os
import tempfile
from functools import lru_cache
from bs4 import BeautifulSoup
from joblib import Parallel, delayed


@lru_cache(None)
def initialize_nltk():

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

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer(), set(stopwords.words('english'))

def get_nltk_resources():
    return initialize_nltk()

def clean_text_imdb(text):

    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except Exception:

        text = re.sub(r'<[^>]+>', ' ', text)

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    text = text.lower()

    keep_words = {'not', 'no', 'never', 'nor', 'neither', 'none', 'cannot', 'without', 'against'}

    lemmatizer, stop_words = get_nltk_resources()
    words = text.split()
    cleaned_words = []
    for word in words:
        if word in keep_words:
            cleaned_words.append(word)
        elif word not in stop_words and len(word) > 2:
            cleaned_words.append(lemmatizer.lemmatize(word))

    return " ".join(cleaned_words)

def preprocess_imdb_data(df):
    df = df.dropna(subset=['review', 'sentiment'])
    df = df.drop_duplicates(subset=['review'])

    print(f"Cleaning {len(df)} reviews in parallel...")
    num_cores = os.cpu_count() or 1
    df['clean_review'] = Parallel(n_jobs=num_cores)(
        delayed(clean_text_imdb)(review) for review in df['review']
    )

    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

def clean_text_mbti(text):

    text = re.sub(r'http\S+', '', text)

    types = ['infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp',
             'isfp', 'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj']

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    text = text.lower()

    for t in types:
        text = text.replace(t, '')

    lemmatizer, stop_words = get_nltk_resources()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

def preprocess_mbti_data(df):
    df['clean_posts'] = df['posts'].apply(clean_text_mbti)
    return df
