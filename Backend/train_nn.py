import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from preprocess import clean_text_mbti, preprocess_mbti_data

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from models_def import MBTINet, PyTorchSklearnWrapper

def train_nn():
    start_time = time.time()
    print("=" * 60)
    print("  MBTI Personality Prediction - PyTorch GPU Model")
    print("=" * 60)
    
    # 1. Load & Preprocess
    df = pd.read_csv('../Dataset/mbti_1.csv')
    df = df.sample(min(len(df), 7000), random_state=42)
    df = preprocess_mbti_data(df)
    
    le = LabelEncoder()
    y = le.fit_transform(df['type'])
    num_classes = len(le.classes_)
    
    # 2. Vectorize
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(df['clean_posts']).toarray()
    
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.1, random_state=42)
    
    # 3. Create PyTorch Objects
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    model = MBTINet(X_train.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    print("\nStarting Training on GPU...")
    epochs = 50
    loss_curve = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_curve.append(round(avg_loss, 4))
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    # 5. Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_t).sum().item() / len(y_test_t)
    
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    
    # 6. Save and Move to Models/
    models_dir = '../Models'
    if not os.path.exists(models_dir): os.makedirs(models_dir)
    
    wrapper = PyTorchSklearnWrapper(model, vectorizer, le, device)
    
    print("\n[+] Saving model locally...")
    joblib.dump(wrapper, 'nn_model.pkl')
    
    print(f"[+] Moving model to {models_dir}...")
    import shutil
    shutil.move('nn_model.pkl', os.path.join(models_dir, 'nn_model.pkl'))
    
    metrics = {
        "accuracy": round(accuracy * 100, 2),
        "training_time": round(time.time() - start_time, 1),
        "device": str(device),
        "architecture": "PyTorch Deep MLP (GPU)"
    }
    metrics_path = os.path.join(models_dir, 'nn_metrics.json')
    with open(metrics_path, 'w') as f: json.dump(metrics, f, indent=2)
    print(f"[+] Neural Network trained on GPU and moved to {models_dir}")

if __name__ == "__main__":
    train_nn()
