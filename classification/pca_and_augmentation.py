import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from classification.datasets import Dataset
from classification.utils.audio_student import Feature_vector_DS
from classification.utils.utils import accuracy
from classification.testRF import get_dataset_matrix
import os

def get_dataset_matrix():
    dataset = Dataset()
    classnames = dataset.list_classes()

    fm_dir = "data/feature_matrices/"
    model_dir = "data/models/"
    os.makedirs(fm_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    myds = Feature_vector_DS(dataset, Nft=512, nmel=20, duration=950, shift_pct=0.5, normalize=True)

    featveclen = len(myds["fire", 0])
    nitems = len(myds)
    naudio = dataset.naudio
    nclass = dataset.nclass
    data_aug_factor = 1
    class_ids_aug = np.repeat(classnames, naudio * data_aug_factor)

    X = np.zeros((data_aug_factor * nclass * naudio, featveclen))
    for s in range(data_aug_factor):
        for class_idx, classname in enumerate(classnames):
            for idx in range(naudio):
                featvec = myds[classname, idx]
                X[s * nclass * naudio + class_idx * naudio + idx, :] = featvec

    y = class_ids_aug.copy()
    return X, y, classnames

def run_pca():
    """
    Perform KFold with model_rf using PCA and model_rf2 without PCA,
    then plot both results on the same figure.
    """
    print("Running PCA analysis with L2 norm:")
    X, y, _ = get_dataset_matrix()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
   
    params = {'bootstrap': True, 'max_depth': 4, 'min_samples_leaf': 2,
            'min_samples_split': 4, 'n_estimators': 150}

    model_rf = RandomForestClassifier(**params)
    model_rf2 = RandomForestClassifier(**params)

    PC_start = 1
    PC_end = 27
    accuracy_mean_pca = []
    accuracy_std_pca = []
    accuracy_mean_noPCA = []
    accuracy_std_noPCA = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for n_components in range(PC_start, PC_end):
        fold_accuracies_pca = []
        fold_accuracies_noPCA = []
        for train_index, val_index in kf.split(X_train):
            X_train_kf, X_val = X_train[train_index], X_train[val_index]
            y_train_kf, y_val = y_train[train_index], y[val_index]

            # L2 norm (applied to both models)
            X_train_kf_norm = X_train_kf / np.linalg.norm(X_train_kf, axis=1, keepdims=True)
            X_val_norm = X_val / np.linalg.norm(X_val, axis=1, keepdims=True)

            # Model_rf with PCA
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train_kf_norm)
            X_val_pca = pca.transform(X_val_norm)
            model_rf.fit(X_train_pca, y_train_kf)
            fold_accuracies_pca.append(accuracy(y_val, model_rf.predict(X_val_pca)))

            # Model_rf2 without PCA
            model_rf2.fit(X_train_kf_norm, y_train_kf)
            fold_accuracies_noPCA.append(accuracy(y_val, model_rf2.predict(X_val_norm)))

        accuracy_mean_pca.append(np.mean(fold_accuracies_pca))
        accuracy_std_pca.append(np.std(fold_accuracies_pca))
        accuracy_mean_noPCA.append(np.mean(fold_accuracies_noPCA))
        accuracy_std_noPCA.append(np.std(fold_accuracies_noPCA))

    # Plot both results on the same figure
    plt.plot(range(PC_start, PC_end), accuracy_mean_pca, label="Mean Accuracy (PCA)", color="blue")
    plt.fill_between(range(PC_start, PC_end),
                     [m - s / 2 for m, s in zip(accuracy_mean_pca, accuracy_std_pca)],
                     [m + s / 2 for m, s in zip(accuracy_mean_pca, accuracy_std_pca)],
                     color='blue', alpha=0.3)

    plt.plot(range(PC_start, PC_end), accuracy_mean_noPCA, label="Mean Accuracy (No PCA)", color="green")
    plt.fill_between(range(PC_start, PC_end),
                     [m - s / 2 for m, s in zip(accuracy_mean_noPCA, accuracy_std_noPCA)],
                     [m + s / 2 for m, s in zip(accuracy_mean_noPCA, accuracy_std_noPCA)],
                     color='green', alpha=0.3)

    plt.xlabel("Number of PCA components")
    plt.ylabel("Mean accuracy")
    plt.title("Comparison: PCA vs. No PCA (L2 norm)")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run_pca()