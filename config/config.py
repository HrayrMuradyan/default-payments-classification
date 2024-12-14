from dataclasses import dataclass

@dataclass
class ProjectParams:
    # Random state for the whole notebook
    random_state: int = 42        

    # Test size of 20% is enough (6000 observations)
    test_size: float = 0.2       

    # The main scoring method
    score: str = "f1"

    # Number of Cross Validation Steps
    cv: int = 5

    # Data Path
    data_path: str = "data/UCI_Credit_Card.csv"

    # Target variable
    target_column: str = "payment_made"