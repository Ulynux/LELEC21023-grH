"""
uart-reader.py
ELEC PROJECT - 210x
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import serial
from serial.tools import list_ports

from classification.utils.plots import plot_specgram

import pickle
# model_knn = pickle.load(open('classification/data/models/modeltest.pickle', 'rb')) # Write your path to the model here!
# model_pca = pickle.load(open('classification/data/models/pca.pickle', 'rb')) # Write your path to the model here!

PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 20

dt = np.dtype(np.uint16).newbyteorder("<")


def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX) :])
    else:
        print(line)
        return None


def reader(port=None):
    ser = serial.Serial(port=port, baudrate=115200)
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=2 * N_MELVECS * MELVEC_LENGTH).decode(
                "utf-8",errors="ignore"
            )
            # print(line)
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            buffer_array = np.frombuffer(buffer, dtype=dt)

            yield buffer_array


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="Port for serial communication")
    args = argParser.parse_args()
    print("uart-reader launched...\n")

    if args.port is None:
        print(
            "No port specified, here is a list of serial communication port available"
        )
        print("================")
        port = list(list_ports.comports())
        for p in port:
            print(p.device)
        print("================")
        print("Launch this script with [-p PORT_REF] to access the communication port")

    else:
        input_stream = reader(port=args.port)
        msg_counter = 0
        all_melvecs = []  # Keep a global list of melvecs
        all_labels = []  # Keep a global list of labels
        # print(input_stream)
        all_classes = ["gunshot", "fireworks", "chainsaw", "crackling fire"]
        classe = all_classes[0]
        for melvec in input_stream:
            print(melvec)
            all_melvecs.append(melvec)
            all_labels.append(classe)
            np.save("/home/ulysse/Documents/LELEC21023-grH/mcu/hands_on_main_app/melspectrograms"+str(classe)+".npy", all_melvecs)
            np.save("/home/ulysse/Documents/LELEC21023-grH/mcu/hands_on_main_app/labels"+str(classe)+".npy", all_labels)
            # melvec = melvec/np.linalg.norm(melvec)
            # melvec = melvec.reshape(1, -1)
            # melvec_reduced = model_pca.transform(melvec)
            # proba_knn = model_knn.predict_proba(melvec_reduced)
            # prediction = model_knn.predict(melvec_reduced)
            msg_counter += 1

            print(f"MEL Spectrogram #{msg_counter}")

            # plt.figure()
            # plot_specgram(
            #     melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T,
            #     ax=plt.gca(),
            #     is_mel=True,
            #     title=f"",
            #     xlabel="Mel vector",
            # )
            # # plt.draw()
            # plt.pause(0.001)
            # plt.savefig(f"mcu/hands_on_feature_vectors/mel_spectrogram_fire10.png")
            # # plt.clf()
