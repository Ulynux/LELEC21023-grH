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
train_results_jack_df = pd.concat([df_train_helicopter, df_train_handsaw, df_train_fire, df_train_chainsaw, df_train_birds], ignore_index=True)

df_test_jack_helicopter = pd.read_csv("predictions_helicopter_generalization_jack.csv")
df_test_jack_handsaw = pd.read_csv("predictions_handsaw_generalization_jack.csv")
df_test_jack_fire = pd.read_csv("predictions_fire_generalization_jack.csv")
df_test_jack_chainsaw = pd.read_csv("predictions_chainsaw_generalization_jack.csv")
df_test_jack_birds = pd.read_csv("predictions_birds_generalization_jack.csv")

# Concatenate the DataFrames for the test set
test_results_jack_df = pd.concat([df_test_jack_helicopter, df_test_jack_handsaw, df_test_jack_fire, df_test_jack_chainsaw, df_test_jack_birds], ignore_index=True)

df_train_helicopter_mic = pd.read_csv("predictions_helicopter_generalization.csv")
df_train_handsaw_mic = pd.read_csv("predictions_handsaw_generalization.csv")
df_train_fire_mic = pd.read_csv("predictions_fire_generalization.csv")
df_train_chainsaw_mic = pd.read_csv("predictions_chainsaw_generalization.csv")
df_train_birds_mic = pd.read_csv("predictions_birds_generalization.csv")

# Concatenate the DataFrames for the test set
train_results_df_mic = pd.concat([df_train_helicopter_mic, df_train_handsaw_mic, df_train_fire_mic, df_train_chainsaw_mic, df_train_birds_mic], ignore_index=True)

df_test_fire_mic = pd.read_csv("predictions_fire_generalization_mic.csv")
df_test_handsaw_mic = pd.read_csv("predictions_handsaw_generalization_mic.csv")
df_test_chainsaw_mic = pd.read_csv("predictions_chainsaw_generalization_mic.csv")
df_test_birds_mic = pd.read_csv("predictions_birds_generalization_mic.csv")
df_test_helicopter_mic = pd.read_csv("predictions_helicopter_generalization_mic.csv")

# Concatenate the DataFrames for the test set
test_results_df_mic = pd.concat([df_test_fire_mic, df_test_handsaw_mic, df_test_chainsaw_mic, df_test_birds_mic, df_test_helicopter_mic], ignore_index=True) 

# Compute mean accuracy for the training/validation set (Jack)
train_accuracy_naive_jack = accuracy_score(train_results_jack_df["true_class"], train_results_jack_df["naive_class"]) * 100
train_accuracy_majority_jack = accuracy_score(train_results_jack_df["true_class"], train_results_jack_df["majority_class"]) * 100
train_accuracy_avg_jack = accuracy_score(train_results_jack_df["true_class"], train_results_jack_df["avg_class"]) * 100
train_accuracy_max_likelihood_jack = accuracy_score(train_results_jack_df["true_class"], train_results_jack_df["max_likelihood_class"]) * 100

print(f"Training accuracy (Naive - Jack): {train_accuracy_naive_jack:.2f}%")
print(f"Training accuracy (Majority Voting - Jack): {train_accuracy_majority_jack:.2f}%")
print(f"Training accuracy (Average Feature - Jack): {train_accuracy_avg_jack:.2f}%")
print(f"Training accuracy (Maximum Likelihood - Jack): {train_accuracy_max_likelihood_jack:.2f}%")

# Compute mean accuracy for the test set (Jack)
test_accuracy_naive_jack = accuracy_score(test_results_jack_df["true_class"], test_results_jack_df["naive_class"]) * 100
test_accuracy_majority_jack = accuracy_score(test_results_jack_df["true_class"], test_results_jack_df["majority_class"]) * 100
test_accuracy_avg_jack = accuracy_score(test_results_jack_df["true_class"], test_results_jack_df["avg_class"]) * 100
test_accuracy_max_likelihood_jack = accuracy_score(test_results_jack_df["true_class"], test_results_jack_df["max_likelihood_class"]) * 100

print(f"Test accuracy (Naive - Jack): {test_accuracy_naive_jack:.2f}%")
print(f"Test accuracy (Majority Voting - Jack): {test_accuracy_majority_jack:.2f}%")
print(f"Test accuracy (Average Feature - Jack): {test_accuracy_avg_jack:.2f}%")
print(f"Test accuracy (Maximum Likelihood - Jack): {test_accuracy_max_likelihood_jack:.2f}%")

# Compute mean accuracy for the training/validation set (Mic)
train_accuracy_naive_mic = accuracy_score(train_results_df_mic["true_class"], train_results_df_mic["naive_class"]) * 100
train_accuracy_majority_mic = accuracy_score(train_results_df_mic["true_class"], train_results_df_mic["majority_class"]) * 100
train_accuracy_avg_mic = accuracy_score(train_results_df_mic["true_class"], train_results_df_mic["avg_class"]) * 100
train_accuracy_max_likelihood_mic = accuracy_score(train_results_df_mic["true_class"], train_results_df_mic["max_likelihood_class"]) * 100

print(f"Training accuracy (Naive - Mic): {train_accuracy_naive_mic:.2f}%")
print(f"Training accuracy (Majority Voting - Mic): {train_accuracy_majority_mic:.2f}%")
print(f"Training accuracy (Average Feature - Mic): {train_accuracy_avg_mic:.2f}%")
print(f"Training accuracy (Maximum Likelihood - Mic): {train_accuracy_max_likelihood_mic:.2f}%")

# Compute mean accuracy for the test set (Mic)
test_accuracy_naive_mic = accuracy_score(test_results_df_mic["true_class"], test_results_df_mic["naive_class"]) * 100
test_accuracy_majority_mic = accuracy_score(test_results_df_mic["true_class"], test_results_df_mic["majority_class"]) * 100
test_accuracy_avg_mic = accuracy_score(test_results_df_mic["true_class"], test_results_df_mic["avg_class"]) * 100
test_accuracy_max_likelihood_mic = accuracy_score(test_results_df_mic["true_class"], test_results_df_mic["max_likelihood_class"]) * 100

print(f"Test accuracy (Naive - Mic): {test_accuracy_naive_mic:.2f}%")
print(f"Test accuracy (Majority Voting - Mic): {test_accuracy_majority_mic:.2f}%")
print(f"Test accuracy (Average Feature - Mic): {test_accuracy_avg_mic:.2f}%")
print(f"Test accuracy (Maximum Likelihood - Mic): {test_accuracy_max_likelihood_mic:.2f}%")

# Compute generalization errors for Jack
generalization_error_naive_jack = train_accuracy_naive_jack - test_accuracy_naive_jack
generalization_error_majority_jack = train_accuracy_majority_jack - test_accuracy_majority_jack
generalization_error_avg_jack = train_accuracy_avg_jack - test_accuracy_avg_jack
generalization_error_max_likelihood_jack = train_accuracy_max_likelihood_jack - test_accuracy_max_likelihood_jack

print(f"Generalization error (Naive - Jack): {generalization_error_naive_jack:.2f}%")
print(f"Generalization error (Majority Voting - Jack): {generalization_error_majority_jack:.2f}%")
print(f"Generalization error (Average Feature - Jack): {generalization_error_avg_jack:.2f}%")
print(f"Generalization error (Maximum Likelihood - Jack): {generalization_error_max_likelihood_jack:.2f}%")

# Compute generalization errors for Mic
generalization_error_naive_mic = train_accuracy_naive_mic - test_accuracy_naive_mic
generalization_error_majority_mic = train_accuracy_majority_mic - test_accuracy_majority_mic
generalization_error_avg_mic = train_accuracy_avg_mic - test_accuracy_avg_mic
generalization_error_max_likelihood_mic = train_accuracy_max_likelihood_mic - test_accuracy_max_likelihood_mic

print(f"Generalization error (Naive - Mic): {generalization_error_naive_mic:.2f}%")
print(f"Generalization error (Majority Voting - Mic): {generalization_error_majority_mic:.2f}%")
print(f"Generalization error (Average Feature - Mic): {generalization_error_avg_mic:.2f}%")
print(f"Generalization error (Maximum Likelihood - Mic): {generalization_error_max_likelihood_mic:.2f}%")

# Plot confusion matrix for Maximum Likelihood - Jack (Training)
show_confusion_matrix(
    train_results_jack_df["max_likelihood_class"],
    train_results_jack_df["true_class"],
    CLASSNAMES,
    title="Confusion Matrix (Maximum Likelihood - Jack Training)",
    filename="confusion_matrix_max_likelihood_jack_training.png"
)

# Plot confusion matrix for Naive - Mic (Test)
show_confusion_matrix(
    test_results_df_mic["naive_class"],
    test_results_df_mic["true_class"],
    CLASSNAMES,
    title="Confusion Matrix (Naive - Mic Test)",
    filename="confusion_matrix_naive_mic_test.png"
)
