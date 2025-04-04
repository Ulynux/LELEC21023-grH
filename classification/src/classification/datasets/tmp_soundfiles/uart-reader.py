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
model_rf = pickle.load(open('classification/data/models/best_rf_model.pickle', 'rb')) # Write your path to the model here!
# model_pca = pickle.load(open('classification/data/models/pca_10_components.pickle', 'rb')) # Write your path to the model here!

# model_knn = pickle.load(open('classification/data/models/modeltest.pickle', 'rb')) # Write your path to the model here!

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
                "ascii"
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

        for melvec in input_stream:
  
            # np.save("/home/ulysse/Documents/LELEC21023-grH/mcu/hands_on_feature_vectors/melspectrograms"+str(classe)+".npy", all_melvecs)
            # np.save("/home/ulysse/Documents/LELEC21023-grH/mcu/hands_on_feature_vectors/labels"+str(classe)+".npy", all_labels)
            melvec = melvec/np.linalg.norm(melvec)
            melvec = melvec.reshape(1, -1)
            proba_knn = model_rf.predict_proba(melvec)
            prediction = model_rf.predict(melvec)
            msg_counter += 1

            print(f"MEL Spectrogram #{msg_counter}")
            print(prediction, proba_knn)
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
            # plt.clf()

# firework-001
# firework-011
# firework-021
# firework-031
# firework-191