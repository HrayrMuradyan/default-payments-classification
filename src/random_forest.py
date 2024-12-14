import sys
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cross_val_utils import extract_validation_info
from config.config import ProjectParams

def report_save_rf_classifier(X_train, y_train, preprocessor_path, save_results, save_model):
    # Instantiate random forest classifier
    random_forest = RandomForestClassifier(random_state=ProjectParams.random_state, max_features="sqrt")

    # Load the preprocessor
    with open(preprocessor_path, 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)

    # Create the random forest pipeline
    rf_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", random_forest),
        ]
    )

    # Define the parameter grid
    rf_param_grid = {
        'classifier__n_estimators': [30, 50, 200],  
        'classifier__max_depth': [10, 15, 20],  
        "classifier__class_weight": ['balanced']
    }

    # Define the parameter grid
    rf_param_grid = {
        'classifier__n_estimators': [30],  
        'classifier__max_depth': [10],  
        "classifier__class_weight": ['balanced']
    }

    # Extract the validation info
    rf_grid_search, rf_val_info = extract_validation_info(rf_pipe, rf_param_grid, X_train, y_train)

    # Save the best model parameters and the best model
    best_model_rf = rf_grid_search.best_estimator_
    best_params_rf = rf_grid_search.best_params_

    # Save the results
    pickle.dump(best_model_rf, open(save_model / "best_random_forest_classifier.pickle", "wb"))
    pickle.dump(best_params_rf, open(save_model / "best_random_forest_hyperparameters.pickle", "wb"))

    # Save the results
    rf_val_info.to_csv(save_results / "best_random_forest_classifier_results.csv", index=False)

    print("Random Forest Hyperparameter tuning done. All the results are saved!")
