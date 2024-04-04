import typer
from spectrogram import create_spectrogram, create_combined_spectrogram
from correlation import process_new_data
from visualise import visualise_data
from extract import extract_features

app = typer.Typer()

DEFAULT_AUDIO = 'speech_recordings/test_1_recording_01_0.flac'
DEFAULT_DIRECTORY = 'speech_recordings'

# Default Usage:
# python3 src/main.py extract
@app.command()
def extract(audio_dir: str = DEFAULT_DIRECTORY, output_file: str = 'features.csv'):
    """
    Extract features from audio file and dump into csv.
    """
    extract_features(audio_dir, output_file)

# Default Usage:
# python3 src/main.py spectrogram
@app.command()
def mfcc(audio_dir: str = DEFAULT_DIRECTORY):
    """
    Generates a spectrogram for the given audio file.
    """
    create_spectrogram(audio_dir)

# Default Usage:
# python3 src/main.py spectrogram
@app.command()
def combined_mfcc(audio_dir: str = DEFAULT_DIRECTORY):
    """
    Generates a spectrogram for the given audio file.
    """
    create_combined_spectrogram(audio_dir)

# Default Usage:
# python3 src/main.py new
@app.command()
def correlation(features_path: str = 'features.csv'):
    """
    Processes new data.
    """
    process_new_data(features_path)

# Default Usage:
# python3 src/main.py visualise
@app.command()
def visualise(features_path: str = 'features.csv'):
    """
    Visualises the audio file data.
    """
    visualise_data(features_path)

if __name__ == "__main__":
    app()
