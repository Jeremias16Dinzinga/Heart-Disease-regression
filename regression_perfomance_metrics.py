from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def plot_targets_versus_model(t, y, title):
    plt.plot(range(len(t)), t, color="blue", label="real")
    plt.plot(range(len(y)), y, color="red", label="model")
    plt.xlabel("Time (index)")
    plt.ylabel("Values")
    plt.title(title)
    plt.grid(True)
    plt.show()


def display_performance_metrics(t, y, title):
    print(title)

    mae = mean_absolute_error(t, y)
    print(f"Mean Absolute Error (MAE): {mae}")
    print()
    plot_targets_versus_model(t, y, title)
