"""
final_compute.py
ELEC PROJECT - 210x
"""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CLASSNAMES = ["birds", "chainsaw", "fire", "handsaw", "helicopter"]

def show_confusion_matrix(y_predict, y_true, classnames, title="", filename=""):
    """
    From target labels and prediction arrays, sort them appropriately and plot confusion matrix.
    The arrays can contain either ints or str quantities, as long as classnames contains all the elements present in them.
    """
    plt.figure(figsize=(5,6))
    confmat = confusion_matrix(y_true, y_predict, labels=classnames)
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
    plt.savefig(filename)
    plt.close()

# Read the CSV files
df_helicopter = pd.read_csv("predictions_helicopter.csv")
df_handsaw = pd.read_csv("predictions_handsaw.csv")
df_fire = pd.read_csv("predictions_fire.csv")
df_chainsaw = pd.read_csv("predictions_chainsaw.csv")
df_birds = pd.read_csv("predictions_birds.csv")

# Concatenate the DataFrames
results_df = pd.concat([df_helicopter, df_handsaw, df_fire, df_chainsaw, df_birds], ignore_index=True)

# Compute mean accuracy
mean_accuracy_naive = accuracy_score(results_df["true_class"], results_df["naive_class"])
mean_accuracy_majority = accuracy_score(results_df["true_class"], results_df["majority_class"])
mean_accuracy_avg = accuracy_score(results_df["true_class"], results_df["avg_class"])
mean_accuracy_max_likelihood = accuracy_score(results_df["true_class"], results_df["max_likelihood_class"])

print(f"Mean accuracy (Naive): {mean_accuracy_naive}")
print(f"Mean accuracy (Majority Voting): {mean_accuracy_majority}")
print(f"Mean accuracy (Average Feature): {mean_accuracy_avg}")
print(f"Mean accuracy (Maximum Likelihood): {mean_accuracy_max_likelihood}")

# Compute confusion matrices
confusion_matrix_naive = confusion_matrix(results_df["true_class"], results_df["naive_class"], labels=CLASSNAMES)
confusion_matrix_majority = confusion_matrix(results_df["true_class"], results_df["majority_class"], labels=CLASSNAMES)
confusion_matrix_avg = confusion_matrix(results_df["true_class"], results_df["avg_class"], labels=CLASSNAMES)
confusion_matrix_max_likelihood = confusion_matrix(results_df["true_class"], results_df["max_likelihood_class"], labels=CLASSNAMES)

# Show confusion matrices
show_confusion_matrix(results_df["naive_class"], results_df["true_class"], CLASSNAMES, title="Naive Method", filename="confusion_matrix_naive.png")
show_confusion_matrix(results_df["majority_class"], results_df["true_class"], CLASSNAMES, title="Majority Voting", filename="confusion_matrix_majority.png")
show_confusion_matrix(results_df["avg_class"], results_df["true_class"], CLASSNAMES, title="Average Feature", filename="confusion_matrix_avg.png")
show_confusion_matrix(results_df["max_likelihood_class"], results_df["true_class"], CLASSNAMES, title="Maximum Likelihood", filename="confusion_matrix_max_likelihood.png")