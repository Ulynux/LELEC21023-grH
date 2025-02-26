import os
import soundfile as sf
import sounddevice as sd
import argparse

def play_sound(file_path):
    data, fs = sf.read(file_path)
    sd.play(data, fs)
    sd.wait()  # Wait until the sound has finished playing

if __name__ == "__main__":
    count = 0
    # Iterate over the dataset and count the number of matching files
    for root, dirs, files in os.walk("data/"):
        for file in files:
            if file.endswith(".wav") and "chainsaw" in file:
                count += 1
    print(f"Number of .wav files containing 'chainsaw': {count}")
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--dataset", default="data/", help="Path to the dataset directory")
    args = argParser.parse_args()
    print("play_sounds launched...\n")
    count = 0
    # Iterate over the dataset and play each sound
    for root, dirs, files in os.walk(args.dataset):
    
        for file in files:
            if file.endswith(".wav") and "gun" in file :#and not "orks" in file:  # Ensure only .wav files are processed
                print(count)
                count += 1
                print(f"Processing file: {file}")
                file_path = os.path.join(root, file)
                print(f"Playing sound: {file_path}")
                play_sound(file_path)