import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import svm

## Configurations
train_filename = "./data/cleveland_train.csv"
targets = ["target"]

C_values_search = np.arange(0.8, 1.2, 0.1)

param = [
    {
        'C': C_values_search,
        'kernel': ['rbf']
    }
]

train_dataset = pd.read_csv(train_filename)
x_train = train_dataset.drop(columns=targets)
t_train = train_dataset[targets] # real

gs = GridSearchCV(
    svm.SVR(),
    param,
    scoring='neg_root_mean_squared_error',
    cv=5,
    verbose=True
)

gs.fit(x_train, t_train.squeeze())

# Print the best parameters
print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)
