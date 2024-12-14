import click
import pandas as pd
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import ProjectParams

from src.dummy_classifier import report_save_dummy_classifier
from src.logistic_regression import report_save_log_reg_classifier
from src.decision_tree import report_save_dt_classifier
from src.random_forest import report_save_rf_classifier
from src.svm import report_save_svm_classifier

@click.command()
@click.option('--processed-data-path',
              type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="Path to the training data set.")
@click.option('--preprocessor-path',
              type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
              help="Path to the training data set.")
@click.option('--models-to', 
              type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True), 
              help="The path to save the downloaded dataset")
@click.option('--results-to', 
              type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True), 
              help="The path to save the downloaded dataset")
def main(processed_data_path, preprocessor_path, models_to, results_to):
    processed_data_path = Path(processed_data_path)
    preprocessor_path = Path(preprocessor_path)
    models_to = Path(models_to)
    results_to = Path(results_to)

    # If preprocessor doesn't exist, raise an error
    if not (preprocessor_path.is_file() and preprocessor_path.name == 'preprocessor.pickle'):
        raise ValueError("preprocessor_path doesn't point to a valid 'preprocessor.pickle' file")

    # If the directory does not exist, create them
    if not models_to.exists():
        models_to.mkdir(parents=True, exist_ok=True)

    if not results_to.exists():
        results_to.mkdir(parents=True, exist_ok=True)

    # If any of the values is not a directory, raise a ValueError
    if not (results_to.is_dir() and models_to.is_dir()):
        raise ValueError("The argument results_to or models_to is not a directory!")

    # Read the data
    X_train = pd.read_csv(processed_data_path / "X_train.csv")
    y_train = pd.read_csv(processed_data_path / "y_train.csv").squeeze()

    # Report and save dummy classifier
    report_save_dummy_classifier(X_train, y_train, save_results=results_to, save_model=models_to)

    # Report and save log reg classifier
    report_save_log_reg_classifier(X_train, y_train, preprocessor_path=preprocessor_path, save_results=results_to, save_model=models_to)

    # Report and save decision tree classifier
    report_save_dt_classifier(X_train, y_train, preprocessor_path=preprocessor_path, save_results=results_to, save_model=models_to)

    # Report and save random forest classifier
    report_save_rf_classifier(X_train, y_train, preprocessor_path=preprocessor_path, save_results=results_to, save_model=models_to)

    # Report and save svm classifier
    report_save_svm_classifier(X_train, y_train, preprocessor_path=preprocessor_path, save_results=results_to, save_model=models_to)

if __name__ == '__main__':
    main()  