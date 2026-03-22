from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
import os
import sys
import nltk
import tempfile
from functools import lru_cache

# Ensure current directory and root is in search path to find local modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Models are usually in a directory relative to the project root
ROOT_DIR = os.path.dirname(BASE_DIR)
MODELS_PATH = os.path.join(ROOT_DIR, 'Models')

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from preprocess import clean_text_imdb, clean_text_mbti
from models_def import NumpySklearnWrapper

app = FastAPI(title="Sentiment and MBTI Prediction API")

# Allow CORS for React frontend (useful for local dev and specific origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In Vercel, we can restrict this further if needed
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Helper: Initialize NLTK safely for Serverless ===
@lru_cache(None)
def setup_nltk():
    nltk_data_dir = os.path.join(tempfile.gettempdir(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    
    # Pre-download VADER lexicon if missing
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', download_dir=nltk_data_dir)
    
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

# === Lazy Model Loading ===
class ModelCache:
    ml_model = None
    ml_vec = None
    nn_model = None
    ml_metrics = None
    nn_metrics = None

    @classmethod
    def get_ml_model(cls):
        if cls.ml_model is None:
            ml_model_file = os.path.join(MODELS_PATH, 'ml_model.pkl')
            ml_vec_file = os.path.join(MODELS_PATH, 'ml_vectorizer.pkl')
            if os.path.exists(ml_model_file):
                cls.ml_model = joblib.load(ml_model_file)
                cls.ml_vec = joblib.load(ml_vec_file)
            else:
                raise FileNotFoundError(f"ML model files not found in {MODELS_PATH}")
        return cls.ml_model, cls.ml_vec

    @classmethod
    def get_nn_model(cls):
        if cls.nn_model is None:
            nn_model_file = os.path.join(MODELS_PATH, 'nn_model.pkl')
            if os.path.exists(nn_model_file):
                cls.nn_model = joblib.load(nn_model_file)
            else:
                raise FileNotFoundError(f"NN model file not found in {MODELS_PATH}")
        return cls.nn_model

    @classmethod
    def get_metrics(cls, type='ml'):
        if type == 'ml':
            if cls.ml_metrics is None:
                ml_metrics_file = os.path.join(MODELS_PATH, 'ml_metrics.json')
                if os.path.exists(ml_metrics_file):
                    with open(ml_metrics_file, 'r') as f:
                        cls.ml_metrics = json.load(f)
            return cls.ml_metrics
        elif type == 'nn':
            if cls.nn_metrics is None:
                nn_metrics_file = os.path.join(MODELS_PATH, 'nn_metrics.json')
                if os.path.exists(nn_metrics_file):
                    with open(nn_metrics_file, 'r') as f:
                        cls.nn_metrics = json.load(f)
            return cls.nn_metrics
        return None

class TextInput(BaseModel):
    text: str

@app.get("/api")
@app.get("/api/")
def read_root():
    return {"message": "Serverless ML (IMDB) & NN (MBTI) Prediction API is running"}

# ==================== ML: IMDB Sentiment Analysis ====================

@app.post("/api/predict/ml")
async def predict_ml(input_data: TextInput):
    try:
        model, vec = ModelCache.get_ml_model()
    except Exception as e:
        import traceback
        error_info = {
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "cwd": os.getcwd(),
            "models_path": MODELS_PATH,
            "models_path_exists": os.path.exists(MODELS_PATH),
            "traceback": traceback.format_exc()
        }
        raise HTTPException(status_code=500, detail=f"Model loading error: {error_info}")
    
    cleaned = clean_text_imdb(input_data.text)
    transformed = vec.transform([cleaned])
    prediction = model.predict(transformed)[0]
    
    # Get probability scores
    proba = model.predict_proba(transformed)[0]
    prob_negative = float(proba[0]) * 100
    prob_positive = float(proba[1]) * 100
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    method_used = "Ensemble (LR + SGD + NB)"
    
    # Initialize VADER (lazy load)
    sia = setup_nltk()
    
    # Lexicon-based override for short texts
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
            
    ml_metrics = ModelCache.get_metrics('ml')
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
    try:
        model = ModelCache.get_nn_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")
    
    cleaned = clean_text_mbti(input_data.text)
    prediction = model.predict([cleaned])[0]
    
    nn_metrics = ModelCache.get_metrics('nn')
    return {
        "prediction": prediction,
        "method": "Deep Learning MLP (NumPy Pure Implementation)",
        "accuracy": f"{nn_metrics['accuracy']}%" if nn_metrics else "N/A"
    }

@app.get("/api/metrics/ml")
def get_ml_metrics():
    metrics = ModelCache.get_metrics('ml')
    if metrics is None:
        raise HTTPException(status_code=404, detail="ML metrics not found")
    return metrics

@app.get("/api/metrics/nn")
def get_nn_metrics():
    metrics = ModelCache.get_metrics('nn')
    if metrics is None:
        raise HTTPException(status_code=404, detail="NN metrics not found")
    return metrics

@app.get("/api/dataset-info")
def get_dataset_info():
    return {
        "ml_dataset": {
            "name": "IMDB Dataset of 50K Movie Reviews",
            "source": "Kaggle",
            "url": "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
            "total_reviews": 50000,
            "classes": ["Positive", "Negative"],
            "features": "Movie review text",
            "preparation": "HTML tag removal, URL removal, special character removal, lemmatization, stopword removal, TF-IDF vectorization."
        },
        "nn_dataset": {
            "name": "MBTI Myers-Briggs Type Indicator",
            "source": "Kaggle",
            "url": "https://www.kaggle.com/datasets/datasnaek/mbti-type",
            "total_entries": 8675,
            "classes": ["16 MBTI types"],
            "features": "Posts (Last 50 posts per user)",
            "preparation": "Text cleaning, MBTI type mentions removal, lemmatization, stopword removal, TF-IDF vectorization."
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
