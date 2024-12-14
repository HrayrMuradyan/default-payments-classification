from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import pickle
import shap
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import ProjectParams


def get_shap_importance(best_model, X_test):
    # Get the feature names
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out() 

    # Get the encoded test data
    X_test_enc = pd.DataFrame(
        data=best_model.named_steps['preprocessor'].transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )

    # Create a shap explainer object 
    shap_explainer = shap.TreeExplainer(
        best_model.named_steps['classifier'],
    )

    # Get the explained test data
    test_shap_explained = shap_explainer(X_test_enc)

    return test_shap_explained

def round_shap_values(shap_values, decimals=4):
    shap_values.values = np.round(shap_values.values, decimals=decimals)
    shap_values.base_values = np.round(shap_values.base_values, decimals=decimals)
    shap_values.data = np.round(shap_values.data, decimals=decimals)
    return shap_values

def get_shap_value_for_label(test_shap_explained, y_test, test_observation_index):
    observation_label = y_test.reset_index(drop=True)[test_observation_index]
    shap_values = round_shap_values(test_shap_explained[:, :, observation_label])
    return shap_values

def shap_waterfall_plot(shap_values, test_observation_index, plot_save_path):
    shap.waterfall_plot(shap_values[test_observation_index], show=False)
    plt.tight_layout()
    plt.savefig(plot_save_path / "shap_waterfall_plot.png")
    plt.close()

def shap_force_plot(shap_values, test_observation_index, plot_save_path):
    shap.plots.force(shap_values[test_observation_index], matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig(plot_save_path / "shap_force_plot.png")
    plt.close()

def shap_global_importance_plot(shap_values, plot_save_path):
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(plot_save_path / "shap_global_importance_plot.png")
    plt.close()

def save_shap_importances_plot(test_shap_explained, y_test, test_observation_index, plot_save_path):
    shap_values = get_shap_value_for_label(test_shap_explained, y_test, test_observation_index)
    shap_values_rounded = round_shap_values(shap_values)

    shap_waterfall_plot(shap_values_rounded, test_observation_index, plot_save_path)
    shap_force_plot(shap_values_rounded, test_observation_index, plot_save_path)
    shap_global_importance_plot(shap_values_rounded, plot_save_path)
    

def decision_tree_importance(dt_model_path, results_save_path):
    results_save_path = Path(results_save_path)

    with open(dt_model_path, 'rb') as best_dt_model:
        best_model_dt = pickle.load(best_dt_model)

    feature_names = best_model_dt.named_steps['preprocessor'].get_feature_names_out() 

    # Get the feature importances from the decision tree classifier
    importances = best_model_dt.named_steps['classifier'].feature_importances_ 

    # Get the feature importance dataframe for decision trees
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances }).sort_values(by="Importance", ascending=False)
    
    # Save the feature importances dataframe
    feature_importance.reset_index(drop=True).to_csv(results_save_path / "dt_feature_importances.csv", index=False)


def get_log_reg_importance(log_reg_model_path, results_save_path):
    results_save_path = Path(results_save_path)

    with open(log_reg_model_path, 'rb') as best_log_reg_model:
        best_model_lr = pickle.load(best_log_reg_model)

    feature_names = (best_model_lr.named_steps['preprocessor'].get_feature_names_out()) 

    # Get the coefficients
    coefficients = best_model_lr.named_steps['classifier'].coef_[0] 

    # Create feature importance data frame, sort it by coefficients
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients}).sort_values(by="Coefficient", ascending=False)

    # Save the feature importances dataframe
    feature_importance.reset_index(drop=True).to_csv(results_save_path / "log_reg_feature_importances.csv", index=False)


def plot_permutation_importance(model, X_train, y_train, save=False, save_path=None):
    """
    Plot the permutation importance of features for a given model.

    Parameters:
    model (sklearn model): A fitted machine learning model.
    X_train (pd.DataFrame): The feature matrix (training set).
    y_train (pd.Series): The target variable (training set).

    Returns:
    None. Displays a box plot of permutation importances.
    
    Raises:
    AssertionError: If model is not a fitted sklearn model.
    AssertionError: If X_train is not a pandas DataFrame.
    AssertionError: If y_train is not a pandas Series.
    """
    assert hasattr(model, "predict"), "Model must be a fitted sklearn model."
    assert isinstance(X_train, pd.DataFrame), "X_train must be a pandas DataFrame."
    assert isinstance(y_train, pd.Series), "y_train must be a pandas Series."

    # Calculate permutation importances
    result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=ProjectParams.random_state)
    
    # Sort importances
    perm_sorted_idx = result.importances_mean.argsort()
    
    # Plot boxplot
    plt.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        tick_labels=X_train.columns[perm_sorted_idx],
    )

    plt.xlabel('Permutation feature importance')
    plt.title('Permutation Feature Importance')

    # If save is True
    if save:
        save_path = Path(save_path)
        # Check if the path exists, create it
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        if save_path.is_dir():
            # Define the path where to save the plot
            file_to_save = save_path / "permutation_importance.png"
            plt.savefig(file_to_save)
        else:
            raise ValueError("The argument save_path is not a directory!")
        
        plt.close()
    
    else:    
        plt.show()