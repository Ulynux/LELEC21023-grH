import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def show_confusion_matrices(test_results_df, classnames):
    fig, axes = plt.subplots(1,4, figsize=(20, 10))
    fig.suptitle('Confusion Matrices')

    methods = [
        ("naive_class", "Naive Method", "confusion_matrix_naive_generalization.png"),
        ("majority_class", "Majority Voting", "confusion_matrix_majority_generalization.png"),
        ("avg_class", "Average Feature", "confusion_matrix_avg_generalization.png"),
        ("max_likelihood_class", "Maximum Likelihood", "confusion_matrix_max_likelihood_generalization.png")
    ]

    for ax, (col, title, filename) in zip(axes.flatten(), methods):
        show_confusion_matrix(test_results_df[col], test_results_df["true_class"], classnames, title=title, ax=ax)
        fig.savefig(filename)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("lol2")

    plt.show()

def show_confusion_matrix(y_predict, y_true, classnames, title="", ax=None):
    """
    From target labels and prediction arrays, sort them appropriately and plot confusion matrix.
    The arrays can contain either ints or str quantities, as long as classnames contains all the elements present in them.
    """
    if ax is None:
        ax = plt.gca()
    confmat = confusion_matrix(y_true, y_predict, labels=classnames)
    sns.heatmap(
        confmat.T,
        square=True,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=classnames,
        yticklabels=classnames,
        ax=ax,
    )
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")
    ax.set_title(title)


CLASSNAMES = ["birds", "chainsaw", "fire", "handsaw", "helicopter"]

df_train_helicopter = pd.read_csv("predictions_helicopter.csv")
df_train_handsaw = pd.read_csv("predictions_handsaw.csv")
df_train_fire = pd.read_csv("predictions_fire.csv")
df_train_chainsaw = pd.read_csv("predictions_chainsaw.csv")
df_train_birds = pd.read_csv("predictions_birds.csv")

training_test = pd.concat([df_train_helicopter,df_train_handsaw,df_train_fire,df_train_chainsaw,df_train_birds])

df_test_helicopter = pd.read_csv("predictions_helicopter_generalization.csv")
df_test_handsaw = pd.read_csv("predictions_handsaw_generalization.csv")
df_test_fire = pd.read_csv("predictions_fire_generalization.csv")
df_test_chainsaw = pd.read_csv("predictions_chainsaw_generalization.csv")
df_test_birds = pd.read_csv("predictions_birds_generalization.csv")
test_results_df = pd.concat([
    df_test_helicopter, df_test_handsaw, df_test_fire, df_test_chainsaw, df_test_birds
])

show_confusion_matrices(training_test, CLASSNAMES)