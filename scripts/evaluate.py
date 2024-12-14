import click
import pandas as pd
from pathlib import Path
import pickle
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.evaluation import calc_all_metrics, plot_confusion_matrix

@click.command()
@click.option('--best-model-path', 
              type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True), 
              help="The path to the best trained model")
@click.option('--processed-data-path',
              type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="Path to the training data set.")
@click.option('--results-to', 
              type=click.Path(exists=True, dir_okay=True, file_okay=False, writable=True), 
              help="The path to save the results to")
@click.option('--plots-to', 
              type=click.Path(exists=True, dir_okay=True, file_okay=False, writable=True), 
              help="The path to save the plots to")
def main(best_model_path, processed_data_path, results_to, plots_to):
    # Convert string paths to Path class
    best_model_path = Path(best_model_path)
    processed_data_path = Path(processed_data_path)
    results_to = Path(results_to)
    plots_to = Path(plots_to)

    if not results_to.exists():
        results_to.mkdir(parents=True, exist_ok=True)

    if not plots_to.exists():
        plots_to.mkdir(parents=True, exist_ok=True)

    # Load the best model given the path
    with open(best_model_path, 'rb') as model_file:
        best_model = pickle.load(model_file)

    # Load the test data
    X_test = pd.read_csv(processed_data_path / "X_test.csv")
    y_test = pd.read_csv(processed_data_path / "y_test.csv").squeeze()

    # Get the test predictions on the best performing model
    y_test_predicted = best_model.predict(X_test)

    # Calculate the metrics for the test data
    calc_all_metrics(y_test, y_test_predicted, save=True, save_path=results_to)

    # Save the confusion matrix plot
    plot_confusion_matrix(y_test, y_test_predicted, class_names=None, title="Confusion Matrix for test data", cmap="Blues", save=True, save_path=plots_to)

if __name__ == '__main__':
    main()  