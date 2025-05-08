import os
import soundfile as sf
import sounddevice as sd
import argparse
import time
from collections import defaultdict

def play_sound(file_path):
    try:
        data, fs = sf.read(file_path)
        sd.play(data, fs)
        sd.wait()  # Wait until the sound has finished playing
    except Exception as e:
        print(f"Error playing {file_path}: {e}")

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--dataset", default="data/", help="Path to the dataset directory")
    argParser.add_argument("-p", "--pause", type=int, default=10, help="Pause duration between sounds (in seconds)")
    args = argParser.parse_args()

    file_list = []
    for root, dirs, files in os.walk(args.dataset):
        for file in files:
            if file.endswith(".wav"):  # Assurez-vous de ne prendre que les fichiers .wav
                file_list.append(os.path.join(root, file))
    file_list.sort()
    print
    class_files = defaultdict(list)
    for file_path in file_list:
        # Si le fichier est dans le dossier racine, utilisez "root" comme classe
        class_name = os.path.basename(os.path.dirname(file_path))
        if class_name == args.dataset.strip("/").split("/")[-1]:  # Si c'est le dossier racine
            class_name = "root"
        class_files[class_name].append(file_path)

    print(f"Classified files: {dict(class_files)}")  # Log pour vérifier les fichiers classés

    for class_name, files in class_files.items():
        print(f"Class: {class_name}, Number of files: {len(files)}")
        if class_name.lower() in ["gun", "fire", "fireworks", "root"]:  # Inclure "root" si nécessaire
            print(f"Processing class: {class_name}")
            for file_path in files[:20]:  # Prenez les 20 premiers fichiers par classe
                file_name = os.path.basename(file_path)
                print(f"Playing sound: {file_name}")
                play_sound(file_path)
                time.sleep(args.pause)