import os
import soundfile as sf
import sounddevice as sd
import argparse
import time
from collections import defaultdict
def play_sound(file_path):
    data, fs = sf.read(file_path)
    sd.play(data, fs)
    sd.wait()  # Wait until the sound has finished playing

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--dataset", default="data/", help="Path to the dataset directory")
    args = argParser.parse_args()
    print("play_sounds launched...\n")
    ## print all the files in the dataset
    print("Dataset files:")
    file_list = []
    for root, dirs, files in os.walk(args.dataset):
        for file in files:
            file_list.append(os.path.join(root, file))
    file_list.sort()
    for file in file_list:
        print(file)
        # Iterate over the sorted dataset and play each sound in order
    # Group files by class (assuming class is determined by the parent directory name)
    class_files = defaultdict(list)
    for file_path in file_list:
        class_name = os.path.basename(os.path.dirname(file_path))
        if file_path.endswith(".wav"):  # Ensure only .wav files are processed
            class_files[class_name].append(file_path)

    # Take up to 20 sounds per class and play them
    for class_name, files in class_files.items():
        print(f"Processing class: {class_name}")
        for file_path in sorted(files)[:20]:  # Take the first 20 files per class
            file_name = os.path.basename(file_path)
            print(f"Current file: {file_name}")
            print(f"Playing sound: {file_path}")
            play_sound(file_path)
            time.sleep(5)
