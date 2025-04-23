import pandas as pd
import shap
import matplotlib.pyplot as plt

def get_feature_importance_rf(model, feature_names, top_n=10):
    """
    Zwraca najważniejsze cechy na podstawie Random Forest.
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return importance_df.head(top_n)


def get_feature_importance_xgb(model, feature_names, top_n=10):
    """
    Zwraca najważniejsze cechy na podstawie XGBoost (gain).
    """
    booster = model.get_booster()
    scores = booster.get_score(importance_type='gain')

    importance_df = pd.DataFrame([
        {'Feature': feature_names[int(k[1:])], 'Importance': v}
        for k, v in scores.items()
    ]).sort_values(by='Importance', ascending=False)

    return importance_df.head(top_n)


def compute_shap_values(model, X_sample, feature_names):
    """
    Oblicza wartości SHAP dla wybranego modelu i podzbioru danych.
    """
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    return shap_df.mean().abs().sort_values(ascending=False)