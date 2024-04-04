import os
import opensmile

def extract_features(directory_path, output_path):
    # Initialize openSMILE
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    
    # Check if output CSV exists; if not, create it with header
    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            f.write('')  # Just to ensure the file is created

    # Iterate over all .flac files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.flac'):
            # Construct the full path to the file
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {file_path}...")

            # Extract features
            features = smile.process_file(file_path)
            
            # Print the first few lines of the features DataFrame
            print(features.head())

            # Append features to the CSV file
            # If the file is empty, write header, otherwise append without header
            if os.path.getsize(output_path) > 0:  # File has content
                features.to_csv(output_path, mode='a', header=False, index=False)
            else:  # File is empty
                features.to_csv(output_path, mode='w', header=True, index=False)
