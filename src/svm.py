import sys
import os
import pickle
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cross_val_utils import extract_validation_info
from config.config import ProjectParams

def report_save_svm_classifier(X_train, y_train, preprocessor_path, save_results, save_model):
    # Define SVM
    svm = SVC(random_state=ProjectParams.random_state, kernel='rbf')

    # Load the preprocessor
    with open(preprocessor_path, 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)

    # Define the SVM pipeline
    svm_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", svm),
        ]
    )

    # Define the SVM parameter grid
    param_grid_svm = {
        'classifier__C': [1, 10],
        'classifier__gamma': [0.01, 0.1],
        'classifier__class_weight': ['balanced']
    }

    # Define the SVM parameter grid
    param_grid_svm = {
        'classifier__C': [1],
        'classifier__gamma': [0.01],
        'classifier__class_weight': ['balanced']
    }

    # Extract the validation info
    svm_grid_search, svm_val_info = extract_validation_info(svm_pipe, param_grid_svm, X_train, y_train)

    # Save the best model parameters and the best model
    best_model_svm = svm_grid_search.best_estimator_
    best_params_svm = svm_grid_search.best_params_

    # Save the results
    pickle.dump(best_model_svm, open(save_model / "best_svm_classifier.pickle", "wb"))
    pickle.dump(best_params_svm, open(save_model / "best_svm_hyperparameters.pickle", "wb"))

    # Save the results
    svm_val_info.to_csv(save_results / "best_svm_classifier_results.csv", index=False)

    print("Support Vector Machines Hyperparameter tuning done. All the results are saved!")
