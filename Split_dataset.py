import pandas as pd

## Configurations
dataset_filename = "./data/cleveland_preprocessed.csv"
train_dataset = "./data/cleveland_train.csv"
test_dataset = "./data/cleveland_test.csv"
perc_train = 80/100

dataset = pd.read_csv(dataset_filename)

# Embaralhar os dados
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

train_samples = int(len(dataset) * perc_train)

train = dataset.iloc[:train_samples]
test  = dataset.iloc[train_samples:]

train.to_csv(train_dataset, index=False)
test.to_csv(test_dataset, index=False)

