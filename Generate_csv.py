import pandas as pd

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

dataset = pd.read_csv(
    "./data/cleveland.data",
    header=None,
    names=column_names,
    na_values='?',
)

dataset.to_csv(
    "./data/cleveland.csv",
    index=False,
)
