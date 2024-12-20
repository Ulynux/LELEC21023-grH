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
    plt.figure(figsize=(5, 6))
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

# Read the CSV files for the training/validation set
df_train_helicopter = pd.read_csv("predictions_helicopter.csv")
df_train_handsaw = pd.read_csv("predictions_handsaw.csv")
df_train_fire = pd.read_csv("predictions_fire.csv")
df_train_chainsaw = pd.read_csv("predictions_chainsaw.csv")
df_train_birds = pd.read_csv("predictions_birds.csv")

# Concatenate the DataFrames for the training/validation set
train_results_df = pd.concat([df_train_helicopter, df_train_handsaw, df_train_fire, df_train_chainsaw, df_train_birds], ignore_index=True)

# Compute mean accuracy for the training/validation set
train_accuracy_naive = accuracy_score(train_results_df["true_class"], train_results_df["naive_class"]) * 100
train_accuracy_majority = accuracy_score(train_results_df["true_class"], train_results_df["majority_class"]) * 100
train_accuracy_avg = accuracy_score(train_results_df["true_class"], train_results_df["avg_class"]) * 100
train_accuracy_max_likelihood = accuracy_score(train_results_df["true_class"], train_results_df["max_likelihood_class"]) * 100

print(f"Training accuracy (Naive): {train_accuracy_naive:.2f}%")
print(f"Training accuracy (Majority Voting): {train_accuracy_majority:.2f}%")
print(f"Training accuracy (Average Feature): {train_accuracy_avg:.2f}%")
print(f"Training accuracy (Maximum Likelihood): {train_accuracy_max_likelihood:.2f}%")

# Read the CSV files for the test set
df_test_helicopter = pd.read_csv("predictions_helicopter_generalization.csv")
df_test_handsaw = pd.read_csv("predictions_handsaw_generalization.csv")
df_test_fire = pd.read_csv("predictions_fire_generalization.csv")
df_test_chainsaw = pd.read_csv("predictions_chainsaw_generalization.csv")
df_test_birds = pd.read_csv("predictions_birds_generalization.csv")

# Concatenate the DataFrames for the test set
test_results_df = pd.concat([df_test_helicopter, df_test_handsaw, df_test_fire, df_test_chainsaw, df_test_birds], ignore_index=True)

# Compute mean accuracy for the test set
test_accuracy_naive = accuracy_score(test_results_df["true_class"], test_results_df["naive_class"]) * 100
test_accuracy_majority = accuracy_score(test_results_df["true_class"], test_results_df["majority_class"]) * 100
test_accuracy_avg = accuracy_score(test_results_df["true_class"], test_results_df["avg_class"]) * 100
test_accuracy_max_likelihood = accuracy_score(test_results_df["true_class"], test_results_df["max_likelihood_class"]) * 100

print(f"Test accuracy (Naive): {test_accuracy_naive:.2f}%")
print(f"Test accuracy (Majority Voting): {test_accuracy_majority:.2f}%")
print(f"Test accuracy (Average Feature): {test_accuracy_avg:.2f}%")
print(f"Test accuracy (Maximum Likelihood): {test_accuracy_max_likelihood:.2f}%")

# Compute generalization error
generalization_error_naive = train_accuracy_naive - test_accuracy_naive
generalization_error_majority = train_accuracy_majority - test_accuracy_majority
generalization_error_avg = train_accuracy_avg - test_accuracy_avg
generalization_error_max_likelihood = train_accuracy_max_likelihood - test_accuracy_max_likelihood

print(f"Generalization error (Naive): {generalization_error_naive:.2f}%")
print(f"Generalization error (Majority Voting): {generalization_error_majority:.2f}%")
print(f"Generalization error (Average Feature): {generalization_error_avg:.2f}%")
print(f"Generalization error (Maximum Likelihood): {generalization_error_max_likelihood:.2f}%")

# Compute confusion matrices for the test set
confusion_matrix_naive = confusion_matrix(test_results_df["true_class"], test_results_df["naive_class"], labels=CLASSNAMES)
confusion_matrix_majority = confusion_matrix(test_results_df["true_class"], test_results_df["majority_class"], labels=CLASSNAMES)
confusion_matrix_avg = confusion_matrix(test_results_df["true_class"], test_results_df["avg_class"], labels=CLASSNAMES)
confusion_matrix_max_likelihood = confusion_matrix(test_results_df["true_class"], test_results_df["max_likelihood_class"], labels=CLASSNAMES)

# Show confusion matrices for the test set
show_confusion_matrix(test_results_df["naive_class"], test_results_df["true_class"], CLASSNAMES, title="Naive Method", filename="confusion_matrix_naive_generalization.png")
show_confusion_matrix(test_results_df["majority_class"], test_results_df["true_class"], CLASSNAMES, title="Majority Voting", filename="confusion_matrix_majority_generalization.png")
show_confusion_matrix(test_results_df["avg_class"], test_results_df["true_class"], CLASSNAMES, title="Average Feature", filename="confusion_matrix_avg_generalization.png")
show_confusion_matrix(test_results_df["max_likelihood_class"], test_results_df["true_class"], CLASSNAMES, title="Maximum Likelihood", filename="confusion_matrix_max_likelihood_generalization.png")
