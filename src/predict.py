import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from extract import extract_features  # Assumes this does not include training
from train import AlzheimerNet  # Make sure this just defines the class, not train it

# Extract features and save them to 'validation.csv'
extract_features(directory_path='non_ad_validation', output_path='validation.csv', has_alzheimers=None)

# Load the extracted features from CSV
data = pd.read_csv('validation.csv')

# Assuming the last column is the target and the rest are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Normalize features as was done during training (if applicable)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)

# Load your trained model
model = AlzheimerNet()
model.load_state_dict(torch.load('alzheimer_model.pth'))
model.eval()  # Ensure the model is in evaluation mode

# Perform prediction
with torch.no_grad():
    predictions = model(X_tensor)
    predicted_labels = (predictions > 0.5).float()  # Assuming binary classification

# Add predicted labels to the DataFrame
data['Predicted_Alzheimer'] = predicted_labels.numpy().flatten()  # Convert tensor to numpy and flatten if necessary

# Save the DataFrame with the predicted labels to a new CSV file
data.to_csv('predictions.csv', index=False)