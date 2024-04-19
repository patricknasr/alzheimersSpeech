import pandas as pd
import joblib
from extract import extract_features

extract_features(directory_path='ad_validation', output_path='kval.csv', has_alzheimers=None)

data = pd.read_csv('kval.csv')

# Step 2: Preprocess the data
# Assuming you have a scaler saved from the training process
scaler = joblib.load('scaler.pkl')  # Load the scaler used during the model training
data_scaled = scaler.transform(data)

# Step 3: Load the trained KNN model
model = joblib.load('knn_model.pkl')  # Load the KNN model

# Step 4: Make predictions
predictions = model.predict(data_scaled)

# Optional: Save predictions to CSV
results = pd.DataFrame(predictions, columns=['has_alzheimers'])
results.to_csv('kpredictions.csv', index=False)

# Printing predictions to console (optional)
print(results)
