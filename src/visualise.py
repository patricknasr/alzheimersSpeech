import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualise_data(feature_file_path):
    df = pd.read_csv(feature_file_path)

    # Ensure 'start' and 'end' are in a proper format if they are to be used for plotting
    # For simplicity, let's assume we're using indices as our x-axis for this example

    # Example 1: Line Plot for MFCCs
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['mfcc1_sma3'], label='MFCC 1')
    plt.plot(df.index, df['mfcc2_sma3'], label='MFCC 2')
    plt.plot(df.index, df['mfcc3_sma3'], label='MFCC 3')
    plt.plot(df.index, df['mfcc4_sma3'], label='MFCC 4')
    plt.xlabel('Time or Frame Index')
    plt.ylabel('MFCC Value')
    plt.title('MFCC Features Over Time')
    plt.legend()
    plt.show()

    # Example 2: Heatmap for a subset of features
    # Selecting a subset of features for the heatmap
    features_for_heatmap = df[['Loudness_sma3', 'alphaRatio_sma3', 'hammarbergIndex_sma3', 'slope0-500_sma3']]
    plt.figure(figsize=(10, 8))
    sns.heatmap(features_for_heatmap.T, cmap='YlGnBu')
    plt.title('Feature Values Heatmap')
    plt.xlabel('Time or Frame Index')
    plt.ylabel('Features')
    plt.show()
