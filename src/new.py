import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV data
df = pd.read_csv('features.csv')

features = ['mfcc1_sma3', 'jitterLocal_sma3nz', 'shimmerLocaldB_sma3nz'] 
correlation_matrix = df[features].corr()

fig, ax = plt.subplots(figsize=(10, 6))

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Selected Speech Features')
plt.show()
