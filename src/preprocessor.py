from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path
import pickle


def define_preprocessor(save_path):
    save_path = Path(save_path)

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    # Create the transformers
    binary_transformer = OneHotEncoder(drop='if_binary')
    nominal_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")
    numeric_transformer = StandardScaler()

    # Define the columns for each data type category
    binary_features = ["sex"]
    nominal_features = ["marriage", "education"]
    numeric_features = ["limit_bal", "age", "pay_amt1", "pay_amt2", "pay_amt3",	\
                        "pay_amt4",	"pay_amt5",	"pay_amt6", "avg_delay", "std_delay", "avg_bill", "std_bill"]

    # Create the preprocessor Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("binary", binary_transformer, binary_features),  
            ("nominal", nominal_transformer, nominal_features),  
            ("numeric", numeric_transformer, numeric_features)  
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # Save the preprocessor
    if save_path.is_dir():
        pickle.dump(preprocessor, open(save_path / "preprocessor.pickle", "wb"))
    else:
        raise ValueError("The argument save_path is not a directory!")