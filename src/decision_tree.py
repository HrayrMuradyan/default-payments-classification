import sys
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cross_val_utils import extract_validation_info
from config.config import ProjectParams

def report_save_dt_classifier(X_train, y_train, preprocessor_path, save_results, save_model):
    # Instantiate the decision tree classifier
    dt = DecisionTreeClassifier(random_state=ProjectParams.random_state)

    # Load the preprocessor
    with open(preprocessor_path, 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)

    # Define the decision tree pipeline
    dt_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", dt),
        ]
    )

    # Define the parameter grid for the decision tree
    dt_param_grid = {
        "classifier__max_depth": [2, 3, 5, 7, 9, 12, 20, 30],
        "classifier__class_weight": ['balanced', None]
    }

    # Get the validation info for the decision tree pipeline
    dt_grid_search, dt_val_info = extract_validation_info(dt_pipe, dt_param_grid, X_train, y_train) 

    # Save the best model parameters and the best model
    best_model_dt = dt_grid_search.best_estimator_
    best_params_dt = dt_grid_search.best_params_

    # Save the results
    pickle.dump(best_model_dt, open(save_model / "best_decision_tree_classifier.pickle", "wb"))
    pickle.dump(best_params_dt, open(save_model / "best_decision_tree_hyperparameters.pickle", "wb"))

    # Save the results
    dt_val_info.to_csv(save_results / "best_decision_tree_classifier_results.csv", index=False)

    print("Decision Tree Hyperparameter tuning done. All the results are saved!")
