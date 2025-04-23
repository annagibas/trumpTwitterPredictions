import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance_bar(importance_df, title="Feature Importance", top_n=10, save_path=None):
    """
    Tworzy wykres słupkowy ważności cech i zapisuje go do pliku PNG, jeśli podano ścieżkę.

    Parametry:
    - importance_df: DataFrame z kolumnami 'Feature' i 'Importance'
    - title: tytuł wykresu
    - top_n: liczba najważniejszych cech do pokazania
    - save_path: pełna ścieżka zapisu wykresu (.png), np. 'results/plots/rf_importance.png'
    """

    top_features = importance_df.sort_values(by="Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=top_features, palette="Blues_d")
    plt.title(title)
    plt.xlabel("Ważność cechy")
    plt.ylabel("Cechy")
    plt.tight_layout()

    if save_path:
        # Tworzenie katalogu, jeśli nie istnieje
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(save_path)
        print(f"Wykres zapisany do pliku: {save_path}")
        plt.close()
    else:
        plt.show()