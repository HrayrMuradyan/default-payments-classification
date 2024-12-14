import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
import click
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import ProjectParams

@click.command()
@click.option('--save-to', 
              type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True), 
              help="The path to save the downloaded dataset")
def main(save_to):
    """
    Preprocesses and splits the raw data, then saves the processed train and test datasets to the specified directory.

    Parameters:
    -----------
    save_to : str or Path
        The directory path where the processed train and test datasets will be saved.

    Returns:
    --------
    None
    """
    raw_data = pd.read_csv(ProjectParams.data_path)

    # Converting the column names to lowercase and renaming the target variable
    # pay_0 should be pay_1
    raw_data.columns = [col.lower() for col in raw_data.columns]
    raw_data.rename({"default.payment.next.month": ProjectParams.target_column, 
                    "pay_0": "pay_1"}, 
                    axis=1, inplace=True)  

    assert raw_data.isnull().sum().sum() == 0, "Number of null cells in the data is not 0!"

    # Drop the unique identifier column
    columns_to_drop = ["id"]
    raw_data.drop(columns_to_drop, axis=1, inplace=True)

    # Convert sex to binary variable (0 and 1)
    raw_data.sex = raw_data.sex - 1
    raw_data.sex = raw_data.sex.replace({1: "male", 0: "female"})

    train_data, test_data = train_test_split(raw_data, test_size=ProjectParams.test_size, random_state=ProjectParams.random_state)

    # Convert the string to Path
    save_to = Path(save_to)

    # If the path doesn't exist, create it
    if not save_to.exists():
        save_to.mkdir(parents=True, exist_ok=True)

    # If save_to is a directory, not a file save the dataset, if not, raise an error
    if save_to.is_dir():
        train_data.to_csv(save_to / "train_data.csv", index=False)
        test_data.to_csv(save_to / "test_data.csv", index=False)
    else:
        raise ValueError("The argument save_to is not a directory!")

    print(f"The dataset was successfully splitted and saved in the directory: \033[1m{save_to}\033[0m\n")

if __name__ == '__main__':
    main()