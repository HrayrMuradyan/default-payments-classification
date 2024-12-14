import sys
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cross_val_utils import extract_validation_info
from config.config import ProjectParams


def report_save_log_reg_classifier(X_train, y_train, preprocessor_path, save_results, save_model):
    # Define the logistic regression
    log_reg = LogisticRegression(random_state=ProjectParams.random_state, max_iter=5000)

    # Load the preprocessor
    with open(preprocessor_path, 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)

    # Define the LR pipeline with preprocessor
    log_reg_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", log_reg),
        ]
    )

    # Define the parameter grid
    param_grid = {
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__class_weight": ['balanced', None]
    }

    # Get the validation info
    log_reg_grid_search, log_reg_val_info = extract_validation_info(log_reg_pipe, param_grid, X_train, y_train)

    # Save the best model parameters and the best model
    best_model_lr = log_reg_grid_search.best_estimator_
    best_params_lr = log_reg_grid_search.best_params_

    # Save the results
    pickle.dump(best_model_lr, open(save_model / "best_log_reg_classifier.pickle", "wb"))
    pickle.dump(best_params_lr, open(save_model / "best_log_reg_hyperparameters.pickle", "wb"))

    # Save the results
    log_reg_val_info.to_csv(save_results / "best_log_reg_classifier_results.csv", index=False)

    print("Logistic Regression Hyperparameter tuning done. All the results are saved!")
