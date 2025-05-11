import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_feature_importance_rf(model, feature_names, save_path=None, top_n=None):
    """
    Zwraca DataFrame z ważnością cech dla modelu Random Forest.

    Argumenty:
    - model: wytrenowany model RandomForestRegressor
    - feature_names: lista nazw cech
    - save_path: opcjonalna ścieżka do zapisu CSV
    - top_n: jeśli podano, zwraca tylko top N cech
    """

    # Wyciąganie ważności cech
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Normalizacja (opcjonalna, ułatwia porównywanie)
    importance_df['Normalized'] = importance_df['Importance'] / importance_df['Importance'].sum()

    # Ograniczenie do top_n cech (jeśli podano)
    if top_n is not None:
        importance_df = importance_df.head(top_n)

    # Zapis do pliku jeśli wskazano ścieżkę
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        importance_df.to_csv(save_path, index=False)

    return importance_df

def get_feature_importance_xgb(model, feature_names, save_path=None, top_n=None):
    """
    Zwraca DataFrame z ważnością cech dla modelu XGBoost (gain-based).

    Argumenty:
    - model: wytrenowany model XGBRegressor
    - feature_names: lista nazw cech
    - save_path: opcjonalna ścieżka do zapisu CSV
    - top_n: jeśli podano, zwraca tylko top N cech
    """
    booster = model.get_booster()
    score = booster.get_score(importance_type='gain')

    importance_df = pd.DataFrame({
        'Feature': list(score.keys()),
        'Gain': list(score.values())
    })

    feature_map = {f"f{i}": name for i, name in enumerate(feature_names)}
    importance_df['Feature'] = importance_df['Feature'].map(feature_map).fillna(importance_df['Feature'])

    importance_df['Gain_norm'] = importance_df['Gain'] / importance_df['Gain'].sum()
    importance_df = importance_df.sort_values(by='Gain', ascending=False)

    if top_n is not None:
        importance_df = importance_df.head(top_n)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        importance_df.to_csv(save_path, index=False)

    return importance_df

    # Pobieranie metryki ważności
    weight_df = extract_importance('weight')
    gain_df = extract_importance('gain')
    cover_df = extract_importance('cover')

    # Połączenie wszystkiego w jeden DataFrame
    importance_df = weight_df.merge(gain_df, on='Feature', how='outer')
    importance_df = importance_df.merge(cover_df, on='Feature', how='outer')

    # Normalizacja metryk
    for col in ['weight', 'gain', 'cover']:
        if col in importance_df.columns:
            importance_df[f'{col}_norm'] = importance_df[col] / importance_df[col].sum()

    # Mapowanie nazw cech z f0 → 'realna_nazwa'
    real_names = {f"f{i}": name for i, name in enumerate(feature_names)}
    importance_df['Feature'] = importance_df['Feature'].map(real_names).fillna(importance_df['Feature'])

    # Sortowanie po gain jeśli dostępny
    if 'gain' in importance_df.columns:
        importance_df = importance_df.sort_values(by='gain', ascending=False)

    # Zapis do pliku
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        importance_df.to_csv(save_path, index=False)

    # Wykres z metryki gain jeśli jest
    if plot_path and 'gain' in importance_df.columns:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='gain', y='Feature', data=importance_df)
        plt.title('XGBoost Feature Importance (Gain)')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return importance_df

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_feature_importance_rf(model, feature_names, save_path=None, top_n=None):
    """
    Zwraca DataFrame z ważnością cech dla modelu Random Forest.

    Argumenty:
    - model: wytrenowany model RandomForestRegressor
    - feature_names: lista nazw cech
    - save_path: opcjonalna ścieżka do zapisu CSV
    - top_n: jeśli podano, zwraca tylko top N cech
    """

    # Wyciąganie ważności cech
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Normalizacja
    importance_df['Normalized'] = importance_df['Importance'] / importance_df['Importance'].sum()

    if top_n is not None:
        importance_df = importance_df.head(top_n)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        importance_df.to_csv(save_path, index=False)

    return importance_df

def get_feature_importance_xgb(model, feature_names, save_path=None, top_n=None):
    """
    Zwraca DataFrame z ważnością cech dla modelu XGBoost (gain-based).

    Argumenty:
    - model: wytrenowany model XGBRegressor
    - feature_names: lista nazw cech
    - save_path: opcjonalna ścieżka do zapisu CSV
    - top_n: jeśli podano, zwraca tylko top N cech
    """

    booster = model.get_booster()
    score = booster.get_score(importance_type='gain')

    # Utworzenie DataFrame z gain
    importance_df = pd.DataFrame({
        'Feature': list(score.keys()),
        'Importance': list(score.values())
    })

    # Mapowanie nazw cech (np. f0 → 'favorite_count')
    feature_map = {f"f{i}": name for i, name in enumerate(feature_names)}
    importance_df['Feature'] = importance_df['Feature'].map(feature_map).fillna(importance_df['Feature'])

    # Normalizacja
    importance_df['Normalized'] = importance_df['Importance'] / importance_df['Importance'].sum()
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    if top_n is not None:
        importance_df = importance_df.head(top_n)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        importance_df.to_csv(save_path, index=False)

    return importance_df