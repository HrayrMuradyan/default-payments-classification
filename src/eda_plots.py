import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def correlation_matrix_plot(dataframe: pd.DataFrame, num_columns: list[str], save: bool = False, save_path: str = None, file_name: str = "correlation_matrix_plot.png"):
    """
    Plots the correlation matrix for numeric features in a given DataFrame.

    This function takes a pandas DataFrame and a list of numeric column names, calculates the correlation matrix for 
    the specified numeric features, and visualizes it using a heatmap.

    Parameters:
    ----------
    dataframe : pandas.DataFrame
        The input DataFrame containing the data to be analyzed.

    num_columns : list of str
        A list of column names corresponding to numeric features in the DataFrame for which the correlation matrix 
        should be calculated and plotted.

    save : bool, optional, default=False
        Whether to save the heatmap as an image file. If True, the heatmap will be saved to the specified path.

    save_path : str, optional, default=None
        The directory path where the heatmap image should be saved. Ignored if `save` is False.

    Returns:
    -------
    None
        This function does not return any value. It displays a heatmap visualization of the correlation matrix.

    Notes:
    -----
    - The correlation matrix is calculated using Pearson's correlation coefficient by default.
    - The heatmap displays the correlation coefficients annotated on the cells, formatted to two decimal places.
    - Ensure the `num_columns` parameter contains valid numeric column names from the provided DataFrame.

    Example:
    --------
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # Example DataFrame
    >>> data = {
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [2, 3, 4, 5, 6]
    ... }
    >>> df = pd.DataFrame(data)
    >>> 
    >>> correlation_matrix_plot(df, ['A', 'B', 'C'])

    """
    assert isinstance(num_columns, list), "The variable 'num_columns' should be a list."
    assert isinstance(dataframe, pd.DataFrame), "The variable 'dataframe' should be a Pandas dataframe."
    assert all([i in dataframe.columns for i in num_columns]), "All columns in the num_columns should be present in the dataframe columns."
    
    # Get the numeric features
    dataframe_num_portion = dataframe[num_columns]

    # Calculate the correlation matrix
    corr_matrix = dataframe_num_portion.corr()

    # Plot the correlation matrix
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()

    # If save is True
    if save:
        save_path = Path(save_path)
        # Check if the path exists, create it
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        if save_path.is_dir():
            # Define the path where to save the plot
            file_to_save = save_path / file_name
            plt.savefig(file_to_save)
        else:
            raise ValueError("The argument save_to is not a directory!")
    
    else:    
        plt.show()


def target_variable_countplot(dataframe: pd.DataFrame, target_column: str, save: bool = False, save_path: str = None):
    """
    Plots the countplot of the target variable to visualize class distribution.

    Parameters:
    dataframe (pd.DataFrame): The input dataframe containing the target column.
    target_column (str): The name of the target column to plot.
    save (bool, optional, default=False): Whether to save the heatmap as an image file. If True, the heatmap will be saved to the specified path.
    save_path (str, optional, default=None): The directory path where the heatmap image should be saved. Ignored if `save` is False.

    Returns:
    None: Displays the countplot of the target variable.
    
    Raises:
    ValueError: If the target_column is not found in the dataframe.
    """
    assert isinstance(dataframe, pd.DataFrame), "'dataframe' variable should be a Pandas dataframe."
    assert isinstance(target_column, str), "'target_column' variable should be a string."
    assert (target_column in dataframe.columns), f"Target column '{target_column}' not found in the dataframe."
    
    # Plot the target variable countplot
    plt.figure(figsize=(5, 5))
    ax = sns.countplot(data=dataframe, x=target_column, hue=target_column, palette=["lightcoral", "lightgreen"], legend=False)
    
    plt.title("Target Variable Distribution")
    plt.xlabel(target_column.title())
    plt.ylabel("Count")
    
    # Add the number of items in each class above the countplot bars
    for p in ax.patches:
        height = p.get_height()  
        ax.text(p.get_x() + p.get_width() / 2, height + 0.1,  
                f'{int(height)}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()

    # If save is True
    if save:
        save_path = Path(save_path)
        # Check if the path exists, create it
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        if save_path.is_dir():
            # Define the path where to save the plot
            file_to_save = save_path / "target_variable_countplot.png"
            plt.savefig(file_to_save)
        else:
            raise ValueError("The argument save_to is not a directory!")
        
    else:    
        plt.show()


def continuous_feat_target_plots(dataframe: pd.DataFrame, continuous_columns: list[str], target_column: str ="payment_made", save: bool = False, save_path: str = None):
    """
    Generates density plots showing the relationship between continuous features and the target variable.

    Parameters:
    train_data (pd.DataFrame): The input training dataframe containing the continuous features and the target column.
    continuous_columns (list of str): A list of continuous feature columns to plot.
    target_column (str): The target variable column name. Default is "payment_made".
    save (bool, optional, default=False): Whether to save the heatmap as an image file. If True, the heatmap will be saved to the specified path.
    save_path (str, optional, default=None): The directory path where the heatmap image should be saved. Ignored if `save` is False.

    Returns:
    None: Displays the density plots.
    """

    assert isinstance(dataframe, pd.DataFrame), "'dataframe' variable should be a Pandas dataframe."
    assert isinstance(target_column, str), "'target_column' variable should be a string."
    assert (target_column in dataframe.columns), f"Target column '{target_column}' not found in the dataframe."
    assert all([i in dataframe.columns for i in continuous_columns]), "All columns in the 'continuous_columns' should be present in the dataframe columns."
    
    # Number of columns and rows for the subplot grid
    n_cols = 3
    n_rows = int(np.ceil(len(continuous_columns) / n_cols)) 

    # Create a figure with a grid layout for subplots
    fig = plt.figure(figsize=(15, 5 * n_rows))  
    gs = fig.add_gridspec(n_rows, n_cols)  

    # Loop through each continuous column and create density plots
    for i, column in enumerate(continuous_columns):
        row = i // n_cols  
        col = i % n_cols   
        
        ax = fig.add_subplot(gs[row, col])  # Add subplot to the grid
        sns.kdeplot(data=dataframe, x=column, hue=target_column, fill=True, ax=ax, common_norm=False)
        
        # Set specific x-axis limits for certain columns if needed
        if column in ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]:
            plt.xlim(-20000, 60000)
        
        # Set title, x-label, and y-label for each plot
        ax.set_title(f'Density Plot of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Density')
        
    # Adjust layout to prevent overlapping elements
    plt.tight_layout()

    # If save is True
    if save:
        save_path = Path(save_path)
        # Check if the path exists, create it
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        if save_path.is_dir():
            # Define the path where to save the plot
            file_to_save = save_path / "continuous_feat_target_plots.png"
            plt.savefig(file_to_save)
        else:
            raise ValueError("The argument save_to is not a directory!")
        
    else:    
        plt.show()


def cat_feat_target_plots(dataframe: pd.DataFrame, categorical_columns: list[str], target_column: str ="payment_made", save: bool = False, save_path: str = None):
    """
    Generates count plots showing the relationship between categorical features and the target variable.

    Parameters:
    dataframe (pd.DataFrame): The input dataframe containing the categorical features and the target column.
    categorical_columns (list of str): A list of categorical feature columns to plot.
    target_column (str): The target variable column name. Default is "payment_made".
    save (bool, optional, default=False): Whether to save the heatmap as an image file. If True, the heatmap will be saved to the specified path.
    save_path (str, optional, default=None): The directory path where the heatmap image should be saved. Ignored if `save` is False.

    Returns:
    None: Displays the count plots.

    Raises:
    ValueError: If the target_column is not found in the dataframe.
    """
    
    assert isinstance(dataframe, pd.DataFrame), "'dataframe' variable should be a Pandas dataframe."
    assert isinstance(target_column, str), "'target_column' variable should be a string."
    assert (target_column in dataframe.columns), f"Target column '{target_column}' not found in the dataframe."
    assert all([i in dataframe.columns for i in categorical_columns]), "All columns in the 'categorical_columns' should be present in the dataframe columns."
    
    n_cols = 3
    n_rows = (len(categorical_columns) + n_cols - 1) // n_cols  # Calculate the number of rows needed

    # Create a figure and axes for subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()  # Flatten axes array to facilitate iteration

    # Loop through each categorical column and create count plots
    for i, column in enumerate(categorical_columns):
        sns.countplot(data=dataframe, x=column, hue=target_column, ax=axes[i])

        # Set title, x-label, and y-label for each plot
        axes[i].set_title(f'Count Plot of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Count')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout to prevent overlapping elements
    plt.tight_layout()

    # If save is True
    if save:
        save_path = Path(save_path)
        # Check if the path exists, create it
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        if save_path.is_dir():
            # Define the path where to save the plot
            file_to_save = save_path / "cat_feat_target_plots.png"
            plt.savefig(file_to_save)
        else:
            raise ValueError("The argument save_to is not a directory!")
        
    else:    
        plt.show()


def feature_distribution_plots(train_data: pd.DataFrame, categorical_features: list[str], save: bool = False, save_path: str = None):
    """
    Generates bar plots showing the distribution of nominal and ordinal features.

    Parameters:
    train_data (pd.DataFrame): The input training dataframe containing the features.
    categorical_features (list of str): A list of categorical column names.
    save (bool, optional, default=False): Whether to save the heatmap as an image file. If True, the heatmap will be saved to the specified path.
    save_path (str, optional, default=None): The directory path where the heatmap image should be saved. Ignored if `save` is False.

    Returns:
    None: Displays the bar plots.
    
    Raises:
    ValueError: If categorical_features variable's elements are not all strings.
    """
    
    # Assert that nominal_features and ordinal_features are lists of strings
    if not isinstance(categorical_features, list) or not all(isinstance(f, str) for f in categorical_features):
        raise ValueError("'categorical_features' must be a list of strings.")
    
    
    # Combine features and calculate the number of subplots required
    n_features = len(categorical_features)
    n_cols = 2  
    n_rows = (n_features + n_cols - 1) // n_cols  

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()  

    for idx, feature in enumerate(categorical_features):
        # Prepare data
        feat_value_counts = train_data[feature].value_counts()
        df = feat_value_counts.reset_index()
        df.columns = [feature, 'Counts']
        
        # Plot on the respective subplot
        ax = sns.barplot(data=df, x=feature, y='Counts', ax=axes[idx])
        
        # Add labels above bars
        for bar in ax.patches:
            bar_height = bar.get_height()
            if bar_height > 0: 
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    bar_height + 100, 
                    f'{int(bar_height)}',  
                    ha='center',  
                    va='bottom', 
                    fontsize=12,  
                    color='black'  
                )
        
        # Set plot limits, labels, and titles
        ax.set_ylim(df.Counts.min(), df.Counts.max() * 1.1)
        ax.set_title(f'Distribution of {feature} Values', fontsize=12)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Counts', fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    # Turn off unused subplots if n_features is not a multiple of n_cols
    for ax in axes[n_features:]:
        ax.axis('off')

    # Adjust layout to prevent overlapping elements
    plt.tight_layout()

    # If save is True
    if save:
        save_path = Path(save_path)
        # Check if the path exists, create it
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        if save_path.is_dir():
            # Define the path where to save the plot
            file_to_save = save_path / "feature_distribution_plots.png"
            plt.savefig(file_to_save)
        else:
            raise ValueError("The argument save_to is not a directory!")
    else:    
        plt.show()
