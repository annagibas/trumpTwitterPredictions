import pandas as pd

def get_feature_importance_rf(model, feature_names):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    return importance_df.sort_values(by='Importance', ascending=False)

def get_feature_importance_xgb(model, feature_names):
    importances = model.get_booster().get_score(importance_type='weight')
    importance_df = pd.DataFrame([
        {'Feature': k, 'Importance': v}
        for k, v in importances.items()
    ])
    return importance_df.sort_values(by='Importance', ascending=False)
