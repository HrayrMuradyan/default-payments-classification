# Predicting the Default Payments of Credit Card Clients in Taiwan from 2005
Author: Hrayr Muradyan

# About

Credit card default prediction is an important task in financial risk management, aiming to identify whether a credit card client is likely to default on their payments. In other words, the goal is to estimate whether a person will fail to pay their credit card bills or not. Defaulting on credit card payments can lead to significant financial losses for banks and lending institutions, impacting both individual clients and the broader economy. Knowing beforehand about the outcome, financial institutions can make informed decisions regarding loan approvals, credit limits, and tailored interventions aimed at reducing the likelihood of defaults.

The "Default of Credit Card Clients" dataset (Lichman, 2013) provides a rich collection of data gathered from 30,000 credit card clients in Taiwan, covering various demographic, financial, and transactional features. Collected over a specific time frame (April 2005 to September 2005), this dataset contains potentially important factors such as payment history, credit utilization, demographic details, and bill statements.

The primary goal of this predictive task is to build a classification model that can effectively distinguish between clients who are likely to default and those who are not. Additionally, find the features (or factors) that are the most important in the predictions.

This report explores the key characteristics of the dataset, analyzes its features, and develops predictive models to address the classification problem of credit card default prediction.

# Report

The final report can be found: [here]()

# Dependencies
This project requires the following Python packages and versions:

- **ipykernel**: Used for interactive computing in Jupyter notebooks.
- **matplotlib**: A library for creating static, animated, and interactive visualizations in Python.
- **numpy**: A package for numerical computing and handling arrays.
- **pandas**: A powerful data manipulation and analysis library.
- **python**: The programming language required to run the project.
- **scikit-learn**: A library for machine learning algorithms and data mining.
- **seaborn**: A Python visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics.
- **jupyterlab**: An Interactive Development Environment to write, debug, and test code.


For the recent versions of the dependencies, view the [environment file]().

### Installation using environment.yml

To ensure a reproducible environment with exact dependency versions, you can use the `environment.yml` file. Follow these steps to set up the environment:

```bash
conda env create -f environment.yml
```
Then, activate the environment:

```bash
conda activate <your-environment-name>
```

# Usage
The steps below outline how to set up and run the analysis. 

### Step 1: Clone the repository

Using Https:
```bash
git clone https://github.com/HrayrMuradyan/default-payments-classification.git
```

Using SSH
```bash
git clone git@github.com:HrayrMuradyan/default-payments-classification.git
```

### Step 2: Activate the environment

> **Note:** The instructions contained in this section assume the commands are executed in a unix-based shell.
    
1. **Navigate to the root directory of the project**: 
    - In the terminal/command line navigate to the root directory of your local copy of this project.
    ```bash
    cd <repo_directory>
    ```
2. **Activate the conda environment**:

    ```bash
    conda activate <your-environment-name>
    ```

### Step 3: Run the Analysis

Open a terminal and run the following commands in the order provided:

```bash
python scripts/data_splitting.py \
    --save-to="./data/raw/"

python scripts/eda.py \
    --train-data-path="./data/raw/train_data.csv" \
    --plot-to="./results/figures/"

python scripts/feature_engineering.py \
    --raw-data-path="./data/raw/" \
    --data-to="./data/processed/" \
    --plot-to="./results/figures/" \
    --preprocessor-to="./results/models/"

python scripts/model_training.py \
    --processed-data-path="./data/processed/" \
    --preprocessor-path="./results/models/preprocessor.pickle" \
    --models-to="./results/models/" \
    --results-to="./results/tables/"

python scripts/feature_importance.py \
    --processed-data-path="./data/processed/" \
    --model-path="./results/models/" \
    --plots-to="./results/figures/" \
    --results-to="./results/tables/"

python scripts/evaluate.py \
    --best-model-path="./results/models/best_random_forest_classifier.pickle" \
    --processed-data-path="./data/processed/" \
    --results-to="./results/tables/" \
    --plots-to="./results/figures/"

quarto render report/default_payments_classification.qmd --to html
quarto render report/default_payments_classification.qmd --to pdf
```

Alternatively you can open the notebook `notebooks/terminal_commands_notebook.ipynb` and run all the cells in order which will execute all the commands mentioned above.

# LICENSE

The code in this repository is licensed under the MIT license. Refer to the [LICENSE](LICENSE) file for more details.

# References

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
