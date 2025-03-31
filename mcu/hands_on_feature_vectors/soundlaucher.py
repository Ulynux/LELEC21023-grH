import os
import soundfile as sf
import sounddevice as sd
import argparse

def play_sound(file_path):
    data, fs = sf.read(file_path)
    sd.play(data, fs)
    sd.wait()  # Wait until the sound has finished playing

play_sound("classification/src/classification/datasets/soundfiles/background.wav")