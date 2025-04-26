import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_importance_bar(importance_df, title, save_path, top_n=20, metric='Importance'):
    """
    Tworzy wykres słupkowy najważniejszych cech według wskazanej metryki.

    Parametry:
    - importance_df: DataFrame z kolumnami 'Feature' oraz wybraną metryką ('Importance', 'gain', 'weight', itp.)
    - title: tytuł wykresu
    - save_path: ścieżka do pliku, w którym zapisze się wykres
    - top_n: ile najważniejszych cech pokazać (domyślnie 20)
    - metric: według której kolumny sortować ('Importance', 'gain', itd.)
    """

    # Sprawdzamy czy wskazana metryka istnieje w DataFrame
    if metric not in importance_df.columns:
        raise ValueError(f"Metryka '{metric}' nie istnieje w danych. Dostępne kolumny: {importance_df.columns.tolist()}")

    # Wybieramy najważniejsze cechy według wskazanej metryki
    top_features = importance_df.sort_values(by=metric, ascending=False).head(top_n)

    # Upewniamy się, że folder na zapis wykresu istnieje
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metric, y='Feature', data=top_features, palette="Blues_d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Wykres zapisany do pliku: {save_path}")