import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def create_spectrogram(directory_path):
    # Get all .flac files in the directory
    flac_files = [f for f in os.listdir(directory_path) if f.endswith('.flac')]
    
    # Determine the number of subplots needed (you might want to adjust this for a large number of files)
    n_files = len(flac_files)
    n_cols = 2  # For example, 2 columns of subplots
    n_rows = np.ceil(n_files / n_cols).astype(int)
    
    # Create a figure to hold the subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()  # Flatten in case of a single row
    
    # Iterate over files and create each spectrogram
    for i, filename in enumerate(flac_files):
        audio_path = os.path.join(directory_path, filename)
        print(f"Processing {audio_path}...")
        
        # Load the audio file
        y, sr = librosa.load(audio_path)
        
        # Generate the spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        
        # Convert to decibels
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Plotting in the subplot
        ax = axes[i]
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax)
        ax.set_title(f'{filename}')
        ax.axis('tight')
    
    # Adjust layout
    plt.tight_layout()
    
    # Show or save the plot
    plt.show()

def create_combined_spectrogram(directory_path):
    concatenated_audio = np.array([])  # Initialize empty array for concatenated audio
    sr_global = None  # Placeholder for the sampling rate

    # Iterate over all files in the directory and concatenate their audio
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith('.flac'):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {file_path}...")
            # Load audio file
            y, sr = librosa.load(file_path)
            if sr_global is None:
                sr_global = sr  # Set global sampling rate based on the first file
            elif sr != sr_global:
                raise ValueError(f"Inconsistent sampling rates: {sr} does not match global {sr_global}")
            concatenated_audio = np.concatenate((concatenated_audio, y)) if concatenated_audio.size else y

    # Generate the Mel Spectrogram for the concatenated audio
    S = librosa.feature.melspectrogram(y=concatenated_audio, sr=sr_global, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plotting
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr_global, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency Spectrogram of Concatenated Audio')
    plt.tight_layout()
    plt.show()
