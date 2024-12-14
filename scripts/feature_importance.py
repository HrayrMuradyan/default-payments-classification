import pickle
from pathlib import Path
import pandas as pd
import click
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.feature_importances import plot_permutation_importance, \
                                    get_log_reg_importance, \
                                    decision_tree_importance, \
                                    get_shap_importance, \
                                    save_shap_importances_plot
                                    

@click.command()
@click.option('--processed-data-path',
              type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="Path to the training data set.")
@click.option('--model-path', 
              type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True), 
              help="The path to save the downloaded dataset")
@click.option('--plots-to', 
              type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True), 
              help="The path to save the downloaded dataset")
@click.option('--results-to', 
              type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True), 
              help="The path to save the downloaded dataset")
def main(processed_data_path, model_path, plots_to, results_to):
    processed_data_path = Path(processed_data_path)
    model_path = Path(model_path)
    results_to = Path(results_to)
    plots_to = Path(plots_to)

    # If the directory does not exist, create them
    if not results_to.exists():
        results_to.mkdir(parents=True, exist_ok=True)

    if not plots_to.exists():
        plots_to.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise ValueError("model_path doesn't point to a valid model directory")
    
    best_model_path = model_path / "best_random_forest_classifier.pickle"

    # Open the best model
    with open(best_model_path, 'rb') as model_file:
        best_model = pickle.load(model_file)

    # Read the data
    X_train = pd.read_csv(processed_data_path / "X_train.csv")
    y_train = pd.read_csv(processed_data_path / "y_train.csv").squeeze()

    X_test = pd.read_csv(processed_data_path / "X_test.csv")
    y_test = pd.read_csv(processed_data_path / "y_test.csv").squeeze()

    # Save the permutation importance plot
    plot_permutation_importance(best_model, X_train, y_train, save=True, save_path=plots_to)

    log_reg_model_path = model_path / "best_log_reg_classifier.pickle"
    # Get the logistic regression importance
    get_log_reg_importance(log_reg_model_path, results_to)

    dt_model_path = model_path / "best_decision_tree_classifier.pickle"
    # Get the decision tree importance
    decision_tree_importance(dt_model_path, results_to)

    # Get the shap importances for the test data
    test_shap_explained = get_shap_importance(best_model, X_test)

    # Save shap importance plots
    save_shap_importances_plot(test_shap_explained, y_test, test_observation_index=0, plot_save_path=plots_to)


if __name__ == '__main__':
    main()  