import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from regression_perfomance_metrics import display_performance_metrics

## Configurations
train_filename = "./data/cleveland_train.csv"
test_filename = "./data/cleveland_test.csv"
targets = ["target"]
n_neighbors = 15
p = 1 # minkowski

train_dataset = pd.read_csv(train_filename)
test_dataset = pd.read_csv(test_filename)

x_train = train_dataset.drop(columns=targets)
t_train = train_dataset[targets] # real

x_test = test_dataset.drop(columns=targets)
t_test = test_dataset[targets] # real

knn = KNeighborsRegressor(
    n_neighbors,
    p = p,
)
knn.fit(x_train, t_train.squeeze())

y_train = knn.predict(x_train) # model output
y_test = knn.predict(x_test) # model output

display_performance_metrics(t_train, y_train, "Train dataset")
display_performance_metrics(t_test, y_test, "Test dataset")