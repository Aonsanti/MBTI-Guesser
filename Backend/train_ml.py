import pandas as pd
import numpy as np
import joblib
import json
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from preprocess import clean_text_imdb, preprocess_imdb_data

def train_ml():
    start_time = time.time()
    print("=" * 60)
    print("  IMDB Sentiment Analysis - Fast 3-Model Ensemble")
    print("=" * 60)

    models_dir = '../Models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    cache_path = os.path.join(models_dir, 'imdb_cleaned.pkl')
    if os.path.exists(cache_path):
        print("[+] Loading preprocessed data from cache...")
        df = joblib.load(cache_path)
    else:
        print("[*] Preprocessing raw data (first run only)...")
        df = pd.read_csv('../Dataset/IMDB Dataset.csv')
        df = preprocess_imdb_data(df)
        joblib.dump(df, cache_path)

    X = df['clean_review']
    y = df['label']

    tfidf_cache = os.path.join(models_dir, 'imdb_tfidf_cache.pkl')
    if os.path.exists(tfidf_cache):
        print("[+] Loading TF-IDF vectors from cache...")
        cached = joblib.load(tfidf_cache)
        X_vec = cached['X_vec']
        vectorizer = cached['vectorizer']
    else:
        print("[*] Building TF-IDF vectors...")
        vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.90,
            sublinear_tf=True,
        )
        X_vec = vectorizer.fit_transform(X)
        joblib.dump({'X_vec': X_vec, 'vectorizer': vectorizer}, tfidf_cache)

    print(f"  Feature matrix: {X_vec.shape[0]} samples x {X_vec.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.15, random_state=42, stratify=y
    )

    print("\n" + "-" * 60)
    print("  Training 3-Model Ensemble")
    print("-" * 60)

    clf_lr = LogisticRegression(
        C=10, max_iter=1000, solver='lbfgs', random_state=42
    )

    clf_sgd = SGDClassifier(
        loss='modified_huber',
        alpha=1e-4,
        max_iter=100,
        tol=1e-3,
        random_state=43,
        n_jobs=-1,
    )

    clf_nb = MultinomialNB(alpha=0.1)

    models = {
        'Logistic Regression': clf_lr,
        'SGD (Modified Huber)': clf_sgd,
        'Multinomial NB': clf_nb,
    }

    trained_models = {}
    individual_accuracies = {}

    for name, clf in models.items():
        t0 = time.time()
        print(f"\n  -> Training {name}...", end=" ", flush=True)

        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"\n    [!] Training failed for {name}: {e}")

        elapsed_model = time.time() - t0
        y_pred_i = clf.predict(X_test)
        acc_i = accuracy_score(y_test, y_pred_i)
        individual_accuracies[name] = round(acc_i * 100, 2)
        trained_models[name] = clf
        print(f"done in {elapsed_model:.1f}s  ->  {acc_i*100:.2f}%")

    print(f"\n  -> Building Soft Voting Ensemble...", end=" ", flush=True)
    ensemble = VotingClassifier(
        estimators=[
            ('lr', trained_models['Logistic Regression']),
            ('sgd', trained_models['SGD (Modified Huber)']),
            ('nb', trained_models['Multinomial NB']),
        ],
        voting='soft',
    )

    ensemble.estimators_ = [
        trained_models['Logistic Regression'],
        trained_models['SGD (Modified Huber)'],
        trained_models['Multinomial NB'],
    ]

    ensemble.le_ = LabelEncoder().fit(y_train)
    ensemble.classes_ = ensemble.le_.classes_
    print("done")

    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for name, acc in individual_accuracies.items():
        print(f"  {name:.<35} {acc:.2f}%")
    print(f"  {'Ensemble (Soft Voting)':.<35} {accuracy*100:.2f}%")
    print(f"\n  Total Training Time: {elapsed:.1f}s")
    print("=" * 60)

    print("\n[+] Saving model locally...")
    joblib.dump(ensemble, 'ml_model.pkl')
    joblib.dump(vectorizer, 'ml_vectorizer.pkl')

    print(f"[+] Moving models to {models_dir}...")
    import shutil
    shutil.move('ml_model.pkl', os.path.join(models_dir, 'ml_model.pkl'))
    shutil.move('ml_vectorizer.pkl', os.path.join(models_dir, 'ml_vectorizer.pkl'))

    metrics = {
        "accuracy": round(accuracy * 100, 2),
        "method": "Ensemble (LR + SGD + NB)",
        "training_time": round(elapsed, 1),
        "individual_models": individual_accuracies,
    }
    metrics_path = os.path.join(models_dir, 'ml_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[+] Model and metrics moved to {models_dir}/")

if __name__ == "__main__":
    train_ml()