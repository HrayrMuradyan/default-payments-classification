import pandas as pd
import numpy as np
import click
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eda_plots import correlation_matrix_plot
from src.preprocessor import define_preprocessor
from config.config import ProjectParams


@click.command()
@click.option('--raw-data-path',
              type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="Path to the training data set.")
@click.option('--data-to', 
              type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True), 
              help="The path to save the downloaded dataset")
@click.option('--plot-to',
              type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True),
              help="Path to directory where the plots from the eda will be saved to.")
@click.option('--preprocessor-to',
              type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True),
              help="Path to directory where the plots from the eda will be saved to.")
def main(raw_data_path, data_to, plot_to, preprocessor_to):
    # Convert the paths to Path classes
    data_path = Path(raw_data_path)
    data_to = Path(data_to)
    plot_to = Path(plot_to)
    preprocessor_to = Path(preprocessor_to)

    # If the directories do not exist, create them
    if not data_to.exists():
        data_to.mkdir(parents=True, exist_ok=True)

    if not plot_to.exists():
        plot_to.mkdir(parents=True, exist_ok=True)

    if not preprocessor_to.exists():
        preprocessor_to.mkdir(parents=True, exist_ok=True)

    # Read the datasets
    train_data = pd.read_csv(data_path / "train_data.csv")
    test_data = pd.read_csv(data_path / "test_data.csv")

    # Apply the transformation to train data
    train_data["education"] = train_data["education"].replace({
        4:"other",
        5:"other",
        6:"other",
        0:"other",
        3:"high_school",
        2:"university",
        1:"graduate"
    })

    # Apply the transformation to test data in the same way
    test_data["education"] = test_data["education"].replace({
        4:"other",
        5:"other",
        6:"other",
        0:"other",
        3:"high_school",
        2:"university",
        1:"graduate"
    })

    # Transform marriage on train data
    train_data["marriage"] = train_data["marriage"].replace({0:"other", 
                                                             1:"married", 
                                                             2:"single", 
                                                             3:"other"})

    # Transform marriage on test data in the same way
    test_data["marriage"] = test_data["marriage"].replace({0:"other", 
                                                           1:"married", 
                                                           2:"single", 
                                                           3:"other"})
    
    # Convert the pay_1 to pay_6 to the avg_delay variable and calculate the standard deviation for the train data
    train_data["avg_delay"] = np.mean([train_data["pay_1"],
                                train_data["pay_2"],
                                train_data["pay_3"],
                                train_data["pay_4"],
                                train_data["pay_5"],
                                train_data["pay_6"]], axis=0)

    train_data["std_delay"] = np.std([train_data["pay_1"], 
                                    train_data["pay_2"], 
                                    train_data["pay_3"], 
                                    train_data["pay_4"], 
                                    train_data["pay_5"], 
                                    train_data["pay_6"]], axis=0)

    train_data.drop(["pay_1", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"], axis=1, inplace=True)

    # Convert the pay_1 to pay_6 to the avg_delay variable and calculate the standard deviation for the test data
    test_data["avg_delay"] = np.mean([test_data["pay_1"],
                                test_data["pay_2"],
                                test_data["pay_3"],
                                test_data["pay_4"],
                                test_data["pay_5"],
                                test_data["pay_6"]], axis=0)

    test_data["std_delay"] = np.std([test_data["pay_1"], 
                                    test_data["pay_2"], 
                                    test_data["pay_3"], 
                                    test_data["pay_4"], 
                                    test_data["pay_5"], 
                                    test_data["pay_6"]], axis=0)

    test_data.drop(["pay_1", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"], axis=1, inplace=True)

    # Convert the bill_amt1 to bill_amt6 to the avg_bill variable and calculate the standard deviation for the train data
    train_data["avg_bill"] = np.mean([train_data["bill_amt1"],
                                train_data["bill_amt2"],
                                train_data["bill_amt3"],
                                train_data["bill_amt4"],
                                train_data["bill_amt5"],
                                train_data["bill_amt6"]], axis=0)

    train_data["std_bill"] = np.std([train_data["bill_amt1"], 
                                    train_data["bill_amt2"], 
                                    train_data["bill_amt3"], 
                                    train_data["bill_amt4"], 
                                    train_data["bill_amt5"], 
                                    train_data["bill_amt6"]], axis=0)


    train_data.drop(["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"], axis=1, inplace=True)

    # Convert the bill_amt1 to bill_amt6 to the avg_bill variable and calculate the standard deviation for the test data
    test_data["avg_bill"] = np.mean([test_data["bill_amt1"],
                                test_data["bill_amt2"],
                                test_data["bill_amt3"],
                                test_data["bill_amt4"],
                                test_data["bill_amt5"],
                                test_data["bill_amt6"]], axis=0)

    test_data["std_bill"] = np.std([test_data["bill_amt1"], 
                                    test_data["bill_amt2"], 
                                    test_data["bill_amt3"], 
                                    test_data["bill_amt4"], 
                                    test_data["bill_amt5"], 
                                    test_data["bill_amt6"]], axis=0)

    test_data.drop(["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"], axis=1, inplace=True)

    # Plot the correlation matrix for the transformed features
    correlation_matrix_plot(train_data, 
                            ['limit_bal', 'age', 'pay_amt1',
                            'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6', 
                            'avg_delay', 'std_delay', 'avg_bill', 'std_bill'], 
                            save=True, 
                            save_path=plot_to,
                            file_name="correlation_matrix_plot_processed.png")
    
    # Prepare the X and y for both train and test data
    X_train = train_data.drop(ProjectParams.target_column, axis=1)
    y_train = train_data[[ProjectParams.target_column]]

    X_test = test_data.drop(ProjectParams.target_column, axis=1)
    y_test = test_data[[ProjectParams.target_column]]

    # If data path is a directory save the data
    if data_to.is_dir():
        X_train.to_csv(data_to / "X_train.csv", index=False)
        X_test.to_csv(data_to / "X_test.csv", index=False)
        y_train.to_csv(data_to / "y_train.csv", index=False)
        y_test.to_csv(data_to / "y_test.csv", index=False)
    else:
        raise ValueError("The argument data_to is not a directory!")
    
    # If preprocessor path is a directory save the preprocessor
    if preprocessor_to.is_dir():
        define_preprocessor(preprocessor_to)
    else:
        raise ValueError("The argument preprocessor_to is not a directory!")
    
    print("Feature Engineering done!")

if __name__ == '__main__':
    main()  
    