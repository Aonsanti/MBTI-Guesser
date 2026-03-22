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
