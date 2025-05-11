import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_model_metrics(metrics_df, save_path_rmse=None, save_path_r2=None):
    """
    Tworzy dwa wykresy słupkowe porównujące:
    - RMSE dla każdego modelu
    - R² dla każdego modelu

    Parametry:
    - metrics_df: DataFrame z kolumnami ['Model', 'RMSE', 'R2']
    - save_path_rmse: ścieżka zapisu wykresu RMSE (opcjonalna)
    - save_path_r2: ścieżka zapisu wykresu R² (opcjonalna)
    """

    # RMSE
    plt.figure(figsize=(10, 6))
    sns.barplot(x='RMSE', y='Model', data=metrics_df.sort_values(by='RMSE'), palette='Blues_r')
    plt.title("Porównanie modeli – RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("Model")
    plt.tight_layout()

    if save_path_rmse:
        _save_plot(save_path_rmse)
    else:
        plt.show()

    # R²
    plt.figure(figsize=(10, 6))
    sns.barplot(x='R2', y='Model', data=metrics_df.sort_values(by='R2', ascending=False), palette='Greens_r')
    plt.title("Porównanie modeli – R²")
    plt.xlabel("R²")
    plt.ylabel("Model")
    plt.tight_layout()

    if save_path_r2:
        _save_plot(save_path_r2)
    else:
        plt.show()


def _save_plot(path):
    """
    Pomocnicza funkcja zapisująca wykres do pliku i tworząca folder
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(path, bbox_inches='tight')
    print(f"Wykres zapisany: {path}")
    plt.close()