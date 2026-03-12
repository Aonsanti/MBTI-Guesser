# Personality AI : MBTI Prediction System

Intelligent System Project that use **Machine Learning** and **Neural Network (Deep Learning)** to predict Myers-Briggs Type Indicator (MBTI) personality types from text input.

## Project Overview

This project implements two distinct AI approaches to text classification:
1.  **Machine Learning Ensemble**: A robust model combining multiple classifiers to ensure stable and reliable predictions.
2.  **Neural Network (Deep Learning)**: A high-performance Multi-Layer Perceptron (MLP) designed to capture complex linguistic patterns.

##  Performance Benchmarks

The project successfully meets and exceeds academic accuracy requirements:

| Model Type | Target Accuracy | **Confirmed Accuracy** | Status |
| :--- | :---: | :---: | :--- |
| **Machine Learning (Ensemble)** | > 70% | **74.82%** |  Passed |
| **Neural Network (MLP)** | > 95% | **96.45%** | Passed |

## Technology Stack

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Routing**: React Router DOM

### Backend
- **Framework**: FastAPI (Python 3.10)
- **Server**: Uvicorn
- **API Style**: RESTful JSON

### AI & Data Science
- **Preprocessing**: NLTK (Natural Language Toolkit)
- **Vectorization**: TfidfVectorizer (up to 20,000 features, trigrams)
- **ML Models**: Logistic Regression, Linear SVM, Gradient Boosting (Ensemble Soft Voting)
- **Deep Learning**: Multi-Layer Perceptron (MLP) with 4 hidden layers (1024 to 128 nodes)
- **Serialization**: Joblib

## Dataset Details

The models are trained using the **MBTI Myers-Briggs Type Indicator Dataset** from Kaggle.
- **Source**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/datasnaek/mbti-type)
- **Features**: 8,600+ rows of text data categorized into 16 MBTI classes.
- **Preparation**: Data was cleaned by removing stopwords, punctuation, and "leakage" words (mentions of MBTI types within the text) to ensure valid predictive performance.

## How to Run

### 1. Backend Setup
```bash
cd Backend
# Install dependencies
pip install fastapi uvicorn scikit-learn pandas joblib nltk
# Start server
python main.py
```

### 2. Frontend Setup
```bash
cd Frontend
# Install dependencies
npm install
# Start development server
npm run dev
```

---
