import os
import soundfile as sf
import sounddevice as sd
import argparse
import time
def play_sound(file_path):
    data, fs = sf.read(file_path)
    sd.play(data, fs)
    sd.wait()  # Wait until the sound has finished playing

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--dataset", default="data/", help="Path to the dataset directory")
    args = argParser.parse_args()
    print("play_sounds launched...\n")

    # Iterate over the dataset and play each sound
    for root, dirs, files in os.walk(args.dataset):
    
        for file in files:
            print(f"Current file: {file}")
            if file.endswith(".wav") and "fire" in file:  # Ensure only .wav files are processed
                print(f"Processing file: {file}")
                file_path = os.path.join(root, file)
                print(f"Playing sound: {file_path}")
                play_sound(file_path)
                time.sleep(5)