import numpy as np
import torch
import torch.nn as nn

class MBTINet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MBTINet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class PyTorchSklearnWrapper:
    def __init__(self, model, vectorizer, label_encoder, device):
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.device = device
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, str): X = [X]
            X_vec = self.vectorizer.transform(X).toarray()
            X_tensor = torch.FloatTensor(X_vec).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return self.label_encoder.inverse_transform(predicted.cpu().numpy())

class NumpySklearnWrapper:
    def __init__(self, weights, biases, vectorizer, label_encoder):
        self.W1, self.W2, self.W3, self.W4, self.W5 = weights
        self.b1, self.b2, self.b3, self.b4, self.b5 = biases
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder

    def relu(self, x):
        return np.maximum(0, x)

    def predict(self, X):
        if isinstance(X, str): X = [X]
        X_vec = self.vectorizer.transform(X).toarray()
        
        out = np.dot(X_vec, self.W1) + self.b1
        out = self.relu(out)
        
        out = np.dot(out, self.W2) + self.b2
        out = self.relu(out)
        
        out = np.dot(out, self.W3) + self.b3
        out = self.relu(out)
        
        out = np.dot(out, self.W4) + self.b4
        out = self.relu(out)
        
        out = np.dot(out, self.W5) + self.b5
        
        predicted = np.argmax(out, axis=1)
        return self.label_encoder.inverse_transform(predicted)
