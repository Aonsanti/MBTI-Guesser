from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import os
import sys

# Ensure current directory is in search path to find local modules (preprocess, models_def)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
import nltk
import tempfile
nltk_data_dir = os.path.join(tempfile.gettempdir(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

try:
    # Try finding it in the standard path or the tmp path
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', download_dir=nltk_data_dir)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from preprocess import clean_text_imdb, clean_text_mbti
from models_def import NumpySklearnWrapper
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
sia = SentimentIntensityAnalyzer()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(BASE_DIR, '..', 'Models')

ml_model = None
ml_vec = None
nn_model = None
nn_vec = None
ml_metrics = None
nn_metrics = None

def load_models():
    global ml_model, ml_vec, nn_model, nn_vec, ml_metrics, nn_metrics
    try:
        ml_model_file = os.path.join(models_path, 'ml_model.pkl')
        ml_vec_file = os.path.join(models_path, 'ml_vectorizer.pkl')
        nn_model_file = os.path.join(models_path, 'nn_model.pkl')
        nn_vec_file = os.path.join(models_path, 'nn_vectorizer.pkl')
        ml_metrics_file = os.path.join(models_path, 'ml_metrics.json')
        nn_metrics_file = os.path.join(models_path, 'nn_metrics.json')

        if os.path.exists(ml_model_file):
            ml_model = joblib.load(ml_model_file)
            ml_vec = joblib.load(ml_vec_file)
        
        # Load Neural Network (MBTI)
        if os.path.exists(nn_model_file):
            nn_model = joblib.load(nn_model_file)
            # If it's the modern wrapper, it already has the vectorizer!
            if hasattr(nn_model, 'vectorizer'):
                nn_vec = nn_model.vectorizer
            elif os.path.exists(nn_vec_file):
                nn_vec = joblib.load(nn_vec_file)
        
        if os.path.exists(ml_metrics_file):
            with open(ml_metrics_file, 'r') as f:
                ml_metrics = json.load(f)
        if os.path.exists(nn_metrics_file):
            with open(nn_metrics_file, 'r') as f:
                nn_metrics = json.load(f)
        
        if ml_model and nn_model:
            print(f"Models loaded successfully from {models_path}")
        else:
            print(f"Warning: Some models were not found in {models_path}")
    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

class TextInput(BaseModel):
    text: str

@app.get("/api/")
def read_root():
    return {"message": "ML (IMDB) & NN (MBTI) Prediction API is running"}

# ==================== ML: IMDB Sentiment Analysis ====================

@app.post("/api/predict/ml")
async def predict_ml(input_data: TextInput):
    if ml_model is None or ml_vec is None:
        raise HTTPException(status_code=500, detail="ML model not loaded")
    
    cleaned = clean_text_imdb(input_data.text)
    vec = ml_vec.transform([cleaned])
    prediction = ml_model.predict(vec)[0]
    
    # Get probability scores
    proba = ml_model.predict_proba(vec)[0]
    prob_negative = float(proba[0]) * 100
    prob_positive = float(proba[1]) * 100
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    method_used = "Ensemble (LR + SGD + NB)"
    
    # VADER overrides for out-of-domain short text (profanity, aggressive intent, edge-cases)
    words_count = len(input_data.text.split())
    if words_count < 20:
        comp = sia.polarity_scores(input_data.text)['compound']
        if comp <= -0.4:
            sentiment = "Negative"
            method_used = "Ensemble + VADER Lexicon Override"
            prob_negative = max(prob_negative, abs(comp) * 100)
            prob_positive = 100 - prob_negative
        elif comp >= 0.4:
            sentiment = "Positive"
            method_used = "Ensemble + VADER Lexicon Override"
            prob_positive = max(prob_positive, comp * 100)
            prob_negative = 100 - prob_positive
            
    return {
        "prediction": sentiment,
        "prob_negative": round(prob_negative, 2),
        "prob_positive": round(prob_positive, 2),
        "method": method_used,
        "accuracy": f"{ml_metrics['accuracy']}%" if ml_metrics else "N/A"
    }

# ==================== NN: MBTI Personality Prediction ====================

@app.post("/api/predict/nn")
async def predict_nn(input_data: TextInput):
    if nn_model is None or nn_vec is None:
        raise HTTPException(status_code=500, detail="NN model not loaded")
    
    cleaned = clean_text_mbti(input_data.text)
    prediction = nn_model.predict([cleaned])[0]
    
    return {
        "prediction": prediction,
        "method": "Deep Learning MLP (4 Hidden Layers, 1024-512-256-128)",
        "accuracy": f"{nn_metrics['accuracy']}%" if nn_metrics else "N/A"
    }

@app.get("/api/metrics/ml")
def get_ml_metrics():
    if ml_metrics is None:
        raise HTTPException(status_code=404, detail="ML metrics not found")
    return ml_metrics

@app.get("/api/metrics/nn")
def get_nn_metrics():
    if nn_metrics is None:
        raise HTTPException(status_code=404, detail="NN metrics not found")
    return nn_metrics

@app.get("/api/dataset-info")
def get_dataset_info():
    return {
        "ml_dataset": {
            "name": "IMDB Dataset of 50K Movie Reviews",
            "source": "Kaggle - IMDB Dataset of 50K Movie Reviews",
            "url": "https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews",
            "total_reviews": 50000,
            "classes": ["Positive", "Negative"],
            "features": "Movie review text (raw HTML text from IMDB)",
            "imperfections": "Contains HTML tags (<br />), special characters, inconsistent formatting, varying review lengths, potential duplicate entries.",
            "preparation": "HTML tag removal (BeautifulSoup), URL removal, special character removal, lowercasing, lemmatization (WordNet), stopword removal, TF-IDF vectorization."
        },
        "nn_dataset": {
            "name": "MBTI Myers-Briggs Type Indicator",
            "source": "Kaggle (mbti_1.csv)",
            "url": "https://www.kaggle.com/datasets/datasnaek/mbti-type",
            "total_entries": 8675,
            "classes": ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
                        "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"],
            "features": "Type (16 MBTI types), Posts (Last 50 posts per user)",
            "imperfections": "Unstructured text, URLs included, HTML tags, special characters, MBTI type mentions in text (data leakage).",
            "preparation": "Text cleaning: lowercasing, removing URLs/punctuation, MBTI type removal, lemmatization, stopword removal, TF-IDF vectorization with trigrams."
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
