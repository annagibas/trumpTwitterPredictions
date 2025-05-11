import shap
import os
import matplotlib.pyplot as plt

def plot_shap_summary(model, X_sample, feature_names=None, save_path=None):
    """
    Tworzy wykres SHAP summary plot dla danego modelu i zbioru danych.

    Parametry:
    - model: wytrenowany model (np. XGBoost, Random Forest, itp.)
    - X_sample: dane wejściowe do obliczeń SHAP (np. X_test)
    - feature_names: lista nazw cech (opcjonalnie)
    - save_path: ścieżka zapisu wykresu (.png)
    """

    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, features=X_sample, feature_names=feature_names, show=False)

    if save_path:
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(save_path, bbox_inches="tight")
        print(f"Wykres SHAP zapisany do pliku: {save_path}")
        plt.close()
    else:
        plt.show()