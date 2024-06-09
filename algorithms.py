import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
from torch_geometric.sampler.utils import X


class CNNClassifier(nn.Module):
    def __init__(self, input_channels, num_classes, embedding_dim=None):
        super(CNNClassifier, self).__init__()

        # If embedding_dim is provided, use an embedding layer
        if embedding_dim:
            self.embedding = nn.Embedding(input_channels, embedding_dim)
        else:
            self.embedding = None

        # Define convolutional layers
        self.conv1 = nn.Conv1d(embedding_dim or input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, num_classes)

    def forward(self, x):
        if self.embedding:
            x = self.embedding(x.long().permute(0, 2, 1))

        x = self.conv1(x)
        x = F.relu(x)

        # Add pooling

        x = self.conv2(x)
        x = F.relu(x)

        x = x.max(dim=2)[0]  # Max pooling over time steps
        x = self.dropout(x)  # Add dropout for regularization
        x = self.fc1(x)
        x = F.softmax(x)
        return x

    def fit(self, train_x, train_y, epochs=10, batch_size=32):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters())

        for epoch in range(epochs):
            for i in range(0, train_x.size(0), batch_size):
                batch_X = Variable(train_x[i:i + batch_size])
                batch_y = Variable(train_y[i:i + batch_size])

                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, test_x):
        with torch.no_grad():
            outputs = self.forward(Variable(test_x))
        _, predicted = torch.max(outputs.data, 1)
        return predicted

    def evaluate_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        kf = KFold(n_splits=5)
        k_fold_scores = []
        for train_index, test_index in kf.split(X):
            train_x, test_x = X[train_index], X[test_index]
            train_y, y_test = y[train_index], y[test_index]
            self.fit(train_x, train_y)
            y_pred = self.predict(test_x)
            k_fold_scores.append(accuracy_score(y_test, y_pred))

        k_fold = sum(k_fold_scores) / len(k_fold_scores)

        return acc, f1, recall, k_fold