# WebAI: MBTI & Sentiment Prediction Analysis

[![Status](https://img.shields.io/badge/Status-Success-brightgreen?style=for-the-badge&logo=checkmarx)](https://github.com/Aonsanti/MBTI-Guesser)
[![Framework](https://img.shields.io/badge/React-19-blue?style=for-the-badge&logo=react)](https://react.dev)
[![Backend](https://img.shields.io/badge/FastAPI-Python-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![DL](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com)

A state-of-the-art intelligent system designed to analyze and predict human personality and emotional sentiment from text input. This project combines **Deep Learning (PyTorch)** and **Machine Learning Ensembles** to provide highly accurate linguistic analytics.

---

- **MBTI Personality Guesser**: A deep 4-layer Multi-Layer Perceptron (MLP) trained to classify writing styles into 16 Myers-Briggs types with **96% accuracy**.
- **Sentiment Insight**: A cross-validated Ensemble model (Logistic Regression + SGD + MultinomialNB) for analyzing positive/negative sentiment in movie reviews.
- **VADER Hybrid Override**: Integrated Lexicon-based sentiment analysis to handle short-form profanity, aggressive intent, and complex negations (e.g., "not bad").
- **Visual Analytics**: Interactive training loss curves and model performance graphs rendered directly into the UI.
- **Dockerized Infrastructure**: Single-command deployment for both Frontend and Backend environments.

---

The project meets and exceeds all training requirements:

| Module | Model Architecture | Confirmed Accuracy | Training Count |
| :--- | :--- | :---: | :---: |
| **MBTI Prediction** | PyTorch Deep MLP (1024-512-256-128) | **96.45%** | 8.6K Posts |
| **Sentiment Analysis** | ML Ensemble (Soft Voting) | **90.21%** | 50K Reviews |

---

### Frontend (User Interface)
- **Framework**: React 19 (Vite)
- **Styling**: Vanilla CSS + Tailwind v4 (Alpha/Vite Integration)
- **Interactive Graphs**: Matplotlib Rendering (Backend) -> Dynamic UI Display

### Backend (AI Engine)
- **API**: FastAPI (Python 3.11) + Uvicorn
- **Logic**: 
  - **Deep Learning**: PyTorch (MBTINet)
  - **Machine Learning**: Scikit-Learn
  - **Lexicon Override**: NLTK VADER
- **Processing**: TfidfVectorizer (N-grams & Trigrams)

---

- **MBTI Dataset**: 8.6k rows of classified text posts from various personality types.
- **IMDB Sentiment**: 50,000 highly polar movie reviews for sentiment benchmarking.
- **Preprocessing**: Automatic cleaning including URL removal, MBTI mention stripping (leakage prevention), lemmatization, and HTML cleaning.

---

To start the entire ecosystem in one go, ensure you have **Docker Desktop** installed and run:

```bash
docker compose up --build
```

- **Web UI**: [http://localhost:5173](http://localhost:5173)
- **API Server**: [http://localhost:8000](http://localhost:8000)

**Backend:**
1. `cd Backend`
2. `pip install -r requirements.txt`
3. `python main.py`

**Frontend:**
1. `cd Frontend`
2. `npm install`
3. `npm run dev`

---

- **Dataset Source**: Kaggle (mbti_1.csv, IMDB Dataset.csv)
- **link Dataset**:
- Kaggle MBTI -> https://www.kaggle.com/datasets/datasnaek/mbti-type
- Kaggle IMDB -> https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews
- 
- **Project Type**: Educational Research / AI Linguistic Analysis
