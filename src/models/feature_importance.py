import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_feature_importance_rf(model, feature_names, save_path=None):
    # Wyciągamy ważność cech z modelu Random Forest
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Normalizujemy, żeby suma ważności wynosiła 1
    importance_df['Normalized'] = importance_df['Importance'] / importance_df['Importance'].sum()

    # Tworzymy folder zapisu jeśli trzeba, zapisujemy do pliku
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        importance_df.to_csv(save_path, index=False)

    return importance_df

def get_feature_importance_xgb(model, feature_names, save_path=None, plot_path=None):
    booster = model.get_booster()

    # Pomocnicza funkcja do pobierania różnych metryk ważności
    def extract_importance(importance_type):
        score = booster.get_score(importance_type=importance_type)
        df = pd.DataFrame([
            {'Feature': k, f'{importance_type}': v}
            for k, v in score.items()
        ])
        return df

    # Pobieramy metryki ważności
    weight_df = extract_importance('weight')
    gain_df = extract_importance('gain')
    cover_df = extract_importance('cover')

    # Łączymy wszystko w jeden DataFrame
    importance_df = weight_df.merge(gain_df, on='Feature', how='outer')
    importance_df = importance_df.merge(cover_df, on='Feature', how='outer')

    # Normalizacja metryk
    for col in ['weight', 'gain', 'cover']:
        if col in importance_df.columns:
            importance_df[f'{col}_norm'] = importance_df[col] / importance_df[col].sum()

    # Mapowanie nazw cech z f0 → 'realna_nazwa'
    real_names = {f"f{i}": name for i, name in enumerate(feature_names)}
    importance_df['Feature'] = importance_df['Feature'].map(real_names).fillna(importance_df['Feature'])

    # Sortujemy po gain jeśli dostępny
    if 'gain' in importance_df.columns:
        importance_df = importance_df.sort_values(by='gain', ascending=False)

    # Zapis do pliku jeśli trzeba
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        importance_df.to_csv(save_path, index=False)

    # Tworzymy wykres z metryki gain jeśli jest
    if plot_path and 'gain' in importance_df.columns:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='gain', y='Feature', data=importance_df)
        plt.title('XGBoost Feature Importance (Gain)')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return importance_df
