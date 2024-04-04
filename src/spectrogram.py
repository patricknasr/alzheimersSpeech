import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def create_spectrogram(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Generate the spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    
    # Convert to decibels
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Plotting
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency Spectrogram')
    plt.tight_layout()
    plt.show()