"""
uart-reader.py
ELEC PROJECT - 210x
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import serial
from serial.tools import list_ports
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import pickle
from classification.utils.plots import plot_specgram
import seaborn as sns
from tensorflow.keras.models import load_model

model_rf = load_model('classification/data/models/best_cnn_last.keras')  # Write your path to the model here!
PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 20

CLASSNAMES = ["chainsaw", "fire", "fireworks","gunshot"]
dt = np.dtype(np.uint16).newbyteorder("<")


def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX):])
    else:
        print(line)
        return None


def reader(port=None):
    ser = serial.Serial(port=port, baudrate=115200)
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=2 * N_MELVECS * MELVEC_LENGTH).decode("ascii")
            # print(line)
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            buffer_array = np.frombuffer(buffer, dtype=dt)
            yield buffer_array


def show_confusion_matrix(y_predict, y_true, classnames, title=""):
    """
    From target labels and prediction arrays, sort them appropriately and plot confusion matrix.
    The arrays can contain either ints or str quantities, as long as classnames contains all the elements present in them.
    """
    plt.figure(figsize=(3, 3))
    confmat = confusion_matrix(y_true, y_predict)
    sns.heatmap(
        confmat.T,
        square=True,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=classnames,
        yticklabels=classnames,
        ax=plt.gca(),
    )
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="Port for serial communication")
    args = argParser.parse_args()
    print("uart-reader launched...\n")
    CLASSNAMES = ["chainsaw", "fire", "fireworks","gunshot"]
    if args.port is None:
        print("No port specified, here is a list of serial communication port available")
        print("================")
        port = list(list_ports.comports())
        for p in port:
            print(p.device)
        print("================")
        print("Launch this script with [-p PORT_REF] to access the communication port")

    else:
        input_stream = reader(port=args.port)
        msg_counter = 0
        moving_avg = 0
        threshold = 5
        energy_flag = False
        memory = []
        
        for melvec in input_stream:
            print(msg_counter)
            msg_counter += 1
            melvec = melvec.copy()
            melvec = melvec.astype(np.float64)

            # melvec is a 20x20 array => compute the energy to see if there is a signal
            tmp_moving_avg = np.convolve(melvec.reshape(-1), np.ones(400) / 400, mode='valid')[0] 
            # convolve retourne un tableau de 1 valeur donc on prend le premier élément

            if moving_avg == 0:
                moving_avg = tmp_moving_avg

            # Threshold ajustable mais 5 * le moving average parait OK
            print(f"Moving average: {moving_avg}")
            if tmp_moving_avg > threshold * moving_avg:
                energy_flag = True
                print(f"Energy spike detected. Threshold: {threshold * moving_avg}")
            else:
                # moyenne des 2 moving average
                moving_avg = (moving_avg + tmp_moving_avg) / 2

            if energy_flag: # Mtn que l'on est sur qu'il y a un signal, on peut faire la classification 
                            # sans regarder à la valeur du moving average car on ne va pas regarder 
                            # qu'après on a plus de signal et stopper la classif en plein milieu
                            # de celle-ci et recommencer à chaque fois 
                        
                print(f"Starting classification")
                
                melvec -= np.mean(melvec)
                melvec = melvec / np.linalg.norm(melvec)

                melvec = melvec.reshape(-1)

                proba_rf = model_rf.predict(melvec)  # Use predict instead of predict_proba
                proba_array = np.array(proba_rf)

                memory.append(proba_array)

                # Only predict after 5 inputs
                if len(memory) >= 5:
                    
                    
                    memory_array = np.array(memory)

                    log_likelihood = np.log(memory_array)
                    log_likelihood_sum = np.sum(log_likelihood, axis=0)

                    sorted_indices = np.argsort(log_likelihood_sum)[::-1]  # Sort in descending order
                    most_likely_class_index = sorted_indices[0]
                    second_most_likely_class_index = sorted_indices[1]

                    confidence = log_likelihood_sum[most_likely_class_index] - log_likelihood_sum[second_most_likely_class_index]

                    # threshold sur la confiance de la prédiction
                    
                    confidence_threshold = 0.45  
                    print(f"Majority voting class after 5 inputs: {majority_class}")

                    # On revient à un état où on relance la classification depuis le début
                    # => on clear la mémoire, et on relance le moving average mais on garde les valeurs 
                    # du moving average précédent sinon on perds trop d'infos
                                                
                    energy_flag  = False
                    memory = []
                    
                    if majority_class == "gun":
                        majority_class = "gunshot"
                    majority_class = CLASSNAMES[majority_class-1]
                    
                    if confidence >= confidence_threshold:
                        print(f"Most likely class index: {most_likely_class_index}")
                        print(f"Confidence: {confidence}")
                                            
                    else:
                        print(f"Confidence too low ({confidence}). Not submitting the guess.")
                    