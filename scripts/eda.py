import pandas as pd
import click
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import ProjectParams
from src.eda_plots import correlation_matrix_plot, \
                          target_variable_countplot, \
                          continuous_feat_target_plots, \
                          cat_feat_target_plots, \
                          feature_distribution_plots

                          

@click.command()
@click.option('--train-data-path',
              type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
              help="Path to the training data set.")
@click.option('--plot-to',
              type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True),
              help="Path to directory where the plots from the eda will be saved to.")
def main(train_data_path, plot_to):
    # Read the training data
    train_data = pd.read_csv(train_data_path)

    # Distribute all the features into their corresponding groups

    continuous_features = ["limit_bal", "bill_amt1", "bill_amt2", "bill_amt3",
                        "bill_amt4", "bill_amt5", "bill_amt6", "pay_amt1",
                        "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5",
                        "pay_amt6", "age"]

    binary_features = ["sex"]
    nominal_features = ["marriage"]
    ordinal_features = ["education", "pay_1", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]

    # Check if all features are in the lists
    assert set(train_data.columns).difference(set(continuous_features + \
                                            binary_features + \
                                            nominal_features + \
                                            ordinal_features + \
                                            [ProjectParams.target_column])) == set(), \
    "Not all features are distributed in their corresponding groups"

    # Plot the correlation matrix for continuous and ordinal features.
    correlation_matrix_plot(train_data, continuous_features + ordinal_features, save=True, save_path=plot_to)

    # Plot the target variable distribution
    target_variable_countplot(train_data, target_column=ProjectParams.target_column, save=True, save_path=plot_to)

    # Plots the density of numeric columns separated by the class by color
    continuous_feat_target_plots(train_data, continuous_columns=continuous_features, target_column=ProjectParams.target_column, save=True, save_path=plot_to)

    # Plot the categorical vs. target variable plots
    cat_feat_target_plots(train_data, binary_features + nominal_features + ordinal_features, target_column=ProjectParams.target_column, save=True, save_path=plot_to)

    # Plot feature distribution plot
    feature_distribution_plots(train_data, ordinal_features + nominal_features, save=True, save_path=plot_to)

    print("EDA done!")

if __name__ == '__main__':
    main()         