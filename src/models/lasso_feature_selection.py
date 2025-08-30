import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from src.data.load_data import load_dataset
from src.utils.feature_engineering import prepare_features

def run_lasso(plot_path="results/plots/lasso_importance.png", csv_path="results/feature_importance/lasso_coefficients.csv"):
    """
    Regresja Lasso z automatycznym doborem parametru alfa.
    Tworzy wykres oraz zapisuje współczynniki regresji, pomagając wybrać najistotniejsze cechy.

    Argumenty:
    - plot_path: ścieżka zapisu wykresu słupkowego
    - csv_path: ścieżka zapisu współczynników jako CSV
    """
    print("Dane wczytane pomyślnie. Wykonuję regresję Lasso...")
    df = load_dataset()
    X_train, X_test, y_train, y_test = prepare_features(df)

    # Skalowanie danych – Lasso jest wrażliwe na skalę
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Trening modelu Lasso z krzyżową walidacją
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, y_train)

    # Współczynniki regresji przypisane do nazw cech
    coef = pd.Series(lasso.coef_, index=X_train.columns)
    coef_sorted = coef.sort_values(ascending=False)

    print("\nWspółczynniki Lasso (posortowane malejąco):")
    print(coef_sorted)

    # Wybór cech z niezerowymi współczynnikami
    selected_features = coef[coef != 0].index.tolist()
    dropped_features = coef[coef == 0].index.tolist()

    print(f"\nLiczba cech wybranych przez Lasso: {len(selected_features)}")
    print(f"Liczba cech odrzuconych przez Lasso (coef = 0): {len(dropped_features)}")

    print("\nCecha uznana za istotną, jeśli jej współczynnik regresji ≠ 0.")
    print("Odrzucone cechy (coef = 0) są uznane za nieistotne – Lasso 'wyzerowuje' ich wpływ na wynik.")

    print("\nWybrane cechy:")
    print(selected_features)

    # Zapis współczynników do pliku CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    coef_df = coef.reset_index()
    coef_df.columns = ["Feature", "Coefficient"]
    coef_df.to_csv(csv_path, index=False)
    print(f"\nWspółczynniki zapisane do: {csv_path}")

    # Wykres feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coef_sorted.values, y=coef_sorted.index, palette="Purples_r")
    plt.title("Lasso Feature Importance (Regression Coefficients)")
    plt.xlabel("Wartość współczynnika")
    plt.ylabel("Cecha")
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Wykres zapisany do: {plot_path}")