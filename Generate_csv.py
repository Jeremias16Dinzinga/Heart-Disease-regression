import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib


raw_dataset_filename = "./data/cleveland.data"
clean_dataset_filename = "./data/cleveland.csv"
preprocessed_dataset_filename = "./data/cleveland_preprocessed.csv"

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

def scatter_plot(x, y, x_label, y_label, title):
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()

def display_correlation(x, y, x_title, y_title):
    correlation = x.corr(y)
    print(f"Correlação entre {x_title} e {y_title}: {correlation * 100:.2f}%")

dataset = pd.read_csv(
    raw_dataset_filename,
    header=None,
    names=column_names,
    na_values='?'
)

dataset = dataset.dropna().reset_index(drop=True)

dataset.to_csv(clean_dataset_filename, index=False)

# scatter_plot(
#     dataset["age"],
#     dataset["thalach"],
#     "Idade",
#     "Frequência Cardíaca Máxima",
#     "Idade vs Thalach"
# )

# display_correlation(
#     dataset["age"],
#     dataset["target"],
#     "Idade",
#     "Doença Cardíaca"
# )

X = dataset.drop(columns=["target"])
y = dataset["target"]

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X)

# Salvar scaler
joblib.dump(scaler, "./models/cleveland_scaler.save")

# Aplicar transformação
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

dataset_preprocessed = X_scaled.copy()
dataset_preprocessed["target"] = y.values

# Salvar dataset pré-processado
dataset_preprocessed.to_csv(preprocessed_dataset_filename, index=False)

print("Pipeline finalizado com sucesso!")
print(f"Dataset limpo: {clean_dataset_filename}")
print(f"Dataset pré-processado: {preprocessed_dataset_filename}")
