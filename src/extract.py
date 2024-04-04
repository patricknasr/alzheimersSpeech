import soundfile as sf
import opensmile

def extract_features(audio_path):
    # Initialize openSMILE
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    
    # Extract features
    features = smile.process_file(audio_path)
    
    # The eGeMAPS feature set may not directly include MFCCs or spectrogram. 
    # You would typically select features related to MFCCs from the available set.
    # For a detailed analysis or customization, consider modifying the configuration file directly.
    print(features.head())
    
    # Save features to a CSV file
    features.to_csv('features.csv')

# Example usage
extract_features('speech_recordings/test_1_recording_01_0.flac')
