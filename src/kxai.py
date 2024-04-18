import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Load data from CSV file
data = pd.read_csv('features.csv')

# # Feature indices
# feature_idx_a = 0
# feature_idx_b = 1

# # Get feature names based on indices
# feature_name_a = data.columns[feature_idx_a]
# feature_name_b = data.columns[feature_idx_b]

feature_name_a = 'jitterLocal_sma3nz'
feature_name_b = 'F1frequency_sma3nz'

data_filtered = data[(data[feature_name_a] > 0.1) & (data[feature_name_b] > 0.1)]

# Select columns dynamically based on indices
X = data_filtered[[feature_name_a, feature_name_b]]

# Assuming 'has_alzheimers' is the name of the target column
y = data_filtered['has_alzheimers']

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Create a mesh grid of points (for visualization)
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict class for each point in the mesh grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k')
plt.title('KNN Class Boundaries with 5 Neighbors')
plt.xlabel(feature_name_a)
plt.ylabel(feature_name_b)
plt.show()
