import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class SpeechFeaturesDataset(Dataset):
    def __init__(self, csv_file):
        # Load the dataset
        self.data_frame = pd.read_csv(csv_file)
        
        # Assuming the last column is the target
        X = self.data_frame.iloc[:, :-1]
        y = self.data_frame.iloc[:, -1]
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Convert arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)  # ensure y is 2D

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loaders(csv_file, batch_size=64, test_split=0.2, shuffle=True):
    # Create an instance of the dataset
    dataset = SpeechFeaturesDataset(csv_file)
    
    # Create train/test split
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
 
class AlzheimerNet(nn.Module):
    def __init__(self):
        super(AlzheimerNet, self).__init__()
        self.fc1 = nn.Linear(25, 128)  # 25 features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view_as(outputs))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = outputs.round()
            total += labels.size(0)
            correct += (predicted == labels.view_as(predicted)).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')
    return accuracy

train_loader, test_loader = create_data_loaders('features.csv')
model = AlzheimerNet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, criterion, optimizer, num_epochs=200)
evaluate_model(model, test_loader)
torch.save(model.state_dict(), 'alzheimer_model.pth')
