import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calc_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, data: str = "Test", save=False, save_path: str = None):
    """
    Calculate and return multiple classification evaluation metrics.

    Parameters:
    - y_true: array-like, true labels of the target variable.
    - y_pred: array-like, predicted labels from the model.

    Returns:
    - A list containing the calculated accuracy, precision, recall, and F1-score.
    """
    save_path = Path(save_path)

    accuracy = accuracy_score(y_true, y_pred) 
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    test_results_df = pd.DataFrame({
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1-Score": [f1]
    })
    
    if (save and save_path is not None):
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        if save_path.is_dir():
            test_results_df.to_csv(save_path / f"final_{data.lower()}_results.csv", index=False)
        else:
            raise ValueError("The argument save_path is not a directory!")
    else:
        return (accuracy, precision, recall, f1)


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, title="Confusion Matrix", cmap="Blues", save=False, save_path=None):
    """
    Plots a confusion matrix using matplotlib and seaborn.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        Ground truth (true labels).
    - y_pred: array-like of shape (n_samples,)
        Predicted labels, as returned by a classifier.
    - class_names: list of str, optional
        List of class names to display on the axes. If None, class indices will be used.
    - normalize: bool, default=False
        If True, normalize the confusion matrix.
    - title: str, optional
        Title for the confusion matrix plot.
    - cmap: str, optional
        Colormap for the heatmap. Default is "Blues".

    Returns:
    - cm: Returns the confusion matrix.
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # Define class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    if (save and save_path is not None):
        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        if save_path.is_dir():
            plt.savefig(save_path / "confusion_matrix.png")
        else:
            raise ValueError("The argument save_path is not a directory!")

    else:
        plt.show()
        return cm

   