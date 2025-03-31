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
model_rf = pickle.load(open('classification/data/models/best_rf_model.pickle', 'rb'))  # Write your path to the model here!
model_pca = pickle.load(open('classification/data/models/pca_25_components.pickle', 'rb'))  # Write your path to the model here!
PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 20

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
            print(line)
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
        memory = []
        input_stream = reader(port=args.port)
        msg_counter = 0
        for melvec in input_stream:
            melvec = melvec.copy()
            melvec = melvec.astype(np.float64)
            print(melvec)

            # Normalize the melvec
            melvec -= np.mean(melvec)
            melvec = melvec / np.linalg.norm(melvec)

            # Reshape melvec to 2D before PCA transformation
            melvec = melvec.reshape(1, -1)
            print(melvec.shape)
            # Apply PCA transformation
            melvec = model_pca.transform(melvec)

            # Predict probabilities and class
            proba_rf = model_rf.predict_proba(melvec)
            prediction = model_rf.predict(melvec)
            memory.append(proba_rf)

            if len(memory) > 5:
                memory.pop(0)

            # Convert memory to numpy array
            memory_array = np.array(memory)

            # Naive method
            naive_class = np.argmax(np.mean(memory_array, axis=0))

            # Majority voting
            majority_class = np.bincount(np.argmax(memory_array, axis=2).flatten()).argmax()

            # Average the feature representation
            avg_feature = np.mean(memory_array, axis=0)
            avg_class = np.argmax(avg_feature)

            # Maximum Likelihood
            likelihoods = np.sum(np.log(memory_array), axis=0)
            max_likelihood_class = np.argmax(likelihoods)


            print(f"Naive class: {CLASSNAMES[naive_class]}")
            print(f"Majority voting class: {CLASSNAMES[majority_class]}")
            print(f"Average feature class: {CLASSNAMES[avg_class]}")
            print(f"Maximum Likelihood class: {CLASSNAMES[max_likelihood_class]}")

            # Break after 101 seconds
            msg_counter += 1

        # Convert results to DataFrame
        # Compute mean accuracy
        # mean_accuracy_naive = accuracy_score(results_df["true_class"], results_df["naive_class"])
        # mean_accuracy_majority = accuracy_score(results_df["true_class"], results_df["majority_class"])
        # mean_accuracy_avg = accuracy_score(results_df["true_class"], results_df["avg_class"])
        # mean_accuracy_max_likelihood = accuracy_score(results_df["true_class"], results_df["max_likelihood_class"])

        # print(f"Mean accuracy (Naive): {mean_accuracy_naive}")
        # print(f"Mean accuracy (Majority Voting): {mean_accuracy_majority}")
        # print(f"Mean accuracy (Average Feature): {mean_accuracy_avg}")
        # print(f"Mean accuracy (Maximum Likelihood): {mean_accuracy_max_likelihood}")

        # # Compute confusion matrices
        # confusion_matrix_naive = confusion_matrix(results_df["true_class"], results_df["naive_class"])
        # confusion_matrix_majority = confusion_matrix(results_df["true_class"], results_df["majority_class"])
        # confusion_matrix_avg = confusion_matrix(results_df["true_class"], results_df["avg_class"])
        # confusion_matrix_max_likelihood = confusion_matrix(results_df["true_class"], results_df["max_likelihood_class"])

        # # Show confusion matrices
        # show_confusion_matrix(results_df["naive_class"], results_df["true_class"], CLASSNAMES, title="Naive Method")
        # show_confusion_matrix(results_df["majority_class"], results_df["true_class"], CLASSNAMES, title="Majority Voting")
        # show_confusion_matrix(results_df["avg_class"], results_df["true_class"], CLASSNAMES, title="Average Feature")
        # show_confusion_matrix(results_df["max_likelihood_class"], results_df["true_class"], CLASSNAMES, title="Maximum Likelihood")
