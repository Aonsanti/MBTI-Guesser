import numpy as np

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
