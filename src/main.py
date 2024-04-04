import typer
from spectrogram import create_spectrogram
from new import process_new_data
from visualise import visualise_data
from extract import extract_features

app = typer.Typer()

DEFAULT_AUDIO = 'speech_recordings/test_1_recording_01_0.flac'

@app.command()
def extract(audio_file: str = DEFAULT_AUDIO, output_file: str = 'features.csv'):
    """
    Extract features from audio file and dump into csv.
    """
    extract_features(audio_file, output_file)

@app.command()
def spectrogram(audio_file: str = DEFAULT_AUDIO):
    """
    Generates a spectrogram for the given audio file.
    """
    create_spectrogram(audio_file)

@app.command()
def new(features_path: str = 'features.csv'):
    """
    Processes new data.
    """
    process_new_data(features_path)

@app.command()
def visualise(features_path: str = 'features.csv'):
    """
    Visualises the audio file data.
    """
    visualise_data(features_path)

if __name__ == "__main__":
    app()
