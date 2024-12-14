import sys
import os
import pandas as pd
from sklearn.dummy import DummyClassifier
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cross_val_utils import mean_std_cross_val_scores
from config.config import ProjectParams


def report_save_dummy_classifier(X_train, y_train, save_results, save_model):
    # Define the DummyClassifier
    dummy_clf = DummyClassifier(strategy="most_frequent", random_state=ProjectParams.random_state)

    # Calculate the mean std cross validation scores
    dummy_results = mean_std_cross_val_scores(
        dummy_clf, X_train, y_train, return_train_score=True, scoring=ProjectParams.score
    )

    # Fit dummy classifier
    dummy_clf.fit(X_train, y_train)

    # Convert the results to data frame and save
    dummy_results_df = pd.DataFrame({"dummy": dummy_results}).T
    dummy_results_df.to_csv(save_results / "dummy_classifier_results.csv", index=False)

    # Save the model
    pickle.dump(dummy_clf, open(save_model / "dummy_classifier.pickle", "wb"))

    print("Dummy Classifier fitting done. All the results are saved!")

