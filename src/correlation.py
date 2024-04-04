import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_new_data(features_file_path):
    df = pd.read_csv(features_file_path)

    features = ['F0semitoneFrom27.5Hz_sma3nz', 'jitterLocal_sma3nz', 'shimmerLocaldB_sma3nz'] 
    correlation_matrix = df[features].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Selected Speech Features')
    plt.show()
