import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from itertools import product
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import ProjectParams


def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Calculate the mean and standard deviation of cross-validation scores for a given model.

    Parameters:
    - model: A scikit-learn estimator implementing `fit` and `predict` methods.
    - X_train: A pandas DataFrame or Series representing the feature matrix for training.
    - y_train: A pandas Series or DataFrame representing the target labels for training.
    - **kwargs: Additional keyword arguments passed to the `cross_validate` function, such as `cv`, `scoring`, etc.

    Returns:
    - A pandas Series with formatted mean cross-validation scores and their standard deviations.
    
    Example:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = ...  # Feature matrix and target labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    cv_scores = mean_std_cross_val_scores(model, X_train, y_train, cv=5, scoring='accuracy')
    print(cv_scores)
    ```
    """
    # Assert that the model is a scikit-learn estimator
    assert hasattr(model, "fit") and hasattr(model, "predict"), "Model should be a valid scikit-learn estimator"
    
    # Assert that X_train and y_train are pandas DataFrames or arrays
    assert isinstance(X_train, (pd.DataFrame, pd.Series)) and isinstance(y_train, (pd.Series, pd.DataFrame)), "X_train and y_train should be either pandas DataFrame or Series"
    
    # Assert that the length of X_train and y_train are compatible
    assert len(X_train) == len(y_train), "X_train and y_train must have the same number of samples"
    
    # Calculate the scores using cross_validate
    scores = cross_validate(model, X_train, y_train, **kwargs)

    # Get the mean scores and std
    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()

    # Round and append the components to the list
    out_col = []
    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], std_scores.iloc[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def extract_validation_info(pipe, param_grid, X_train, y_train):
    """
    Extracts validation performance metrics for different parameter combinations 
    using GridSearchCV.

    Parameters:
    - pipe: A scikit-learn pipeline object.
    - param_grid: Dictionary specifying parameter grid to search over.
    - X_train: A pandas DataFrame or Series representing the feature matrix for training.
    - y_train: A pandas Series or DataFrame representing the target labels for training.

    Returns:
    - grid_search: A fitted GridSearchCV object.
    - results_df: A pandas DataFrame with the validation performance metrics sorted by F1 score.

    Raises:
    - AssertionError if `pipe` is not a valid scikit-learn pipeline.
    - AssertionError if `param_grid` is not a dictionary.
    - AssertionError if `X_train` and `y_train` are not pandas DataFrame or Series and have different lengths.
    """

    # Ensure pipe is a valid scikit-learn pipeline
    assert hasattr(pipe, "fit") and hasattr(pipe, "predict"), "'pipe' should be a valid scikit-learn pipeline"
    
    # Ensure param_grid is a dictionary
    assert isinstance(param_grid, dict), "'param_grid' should be a dictionary"
    
    # Ensure X_train and y_train are pandas DataFrame or Series and have the same length
    assert isinstance(X_train, (pd.DataFrame, pd.Series)) and isinstance(y_train, (pd.Series, pd.DataFrame)), \
        "X_train and y_train should be pandas DataFrame or Series"
    assert len(X_train) == len(y_train), "X_train and y_train must have the same number of samples"

    # Define scorers for various metrics
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    
    # Instantiate the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=pipe, 
        param_grid=param_grid, 
        scoring=scorers, 
        refit=ProjectParams.score,  
        cv=ProjectParams.cv, 
        return_train_score=True
    )
    
    # Fit the GridSearchCV
    grid_search.fit(X_train, y_train)

    # Create a list to store results
    results = []
    
    # Get all parameter combinations
    param_combinations = list(product(*param_grid.values()))

    # For each parameter combination
    for combo in param_combinations:
        param_dict = dict(zip(param_grid.keys(), combo))
        
        # Match the current parameter combination with GridSearchCV results
        for i, param_value in enumerate(grid_search.cv_results_['params']):
            if all(param_dict[key] == param_value[key] for key in param_dict):
                # Store the metrics from GridSearch results
                metrics = {
                    **param_dict,
                    'accuracy_train': grid_search.cv_results_['mean_train_accuracy'][i],
                    'accuracy_val': grid_search.cv_results_['mean_test_accuracy'][i],
                    'precision_train': grid_search.cv_results_['mean_train_precision'][i],
                    'precision_val': grid_search.cv_results_['mean_test_precision'][i],
                    'recall_train': grid_search.cv_results_['mean_train_recall'][i],
                    'recall_val': grid_search.cv_results_['mean_test_recall'][i],
                    'f1_train': grid_search.cv_results_['mean_train_f1'][i],
                    'f1_val': grid_search.cv_results_['mean_test_f1'][i],
                    'f1_val_std': grid_search.cv_results_['std_test_f1'][i],
                    'avg_fit_time': grid_search.cv_results_['mean_fit_time'][i],
                    'avg_score_time': grid_search.cv_results_['mean_score_time'][i]
                }
                results.append(metrics)
                break

    # Convert results to DataFrame and sort by F1 validation score
    results_df = pd.DataFrame(results).sort_values(by=f"{ProjectParams.score}_val", ascending=False)
    return grid_search, results_df

