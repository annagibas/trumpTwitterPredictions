import sys
print("Python version:", sys.executable)

from src.data.load_data import load_dataset
from src.utils.feature_engineering import prepare_features
from src.models.train import train_all_models
from src.models.evaluate import compare_models
from src.models.feature_importance import get_feature_importance_rf, get_feature_importance_xgb
from src.visualization.plot_feature_importance import plot_feature_importance_bar
from src.visualization.plot_shap_summary import plot_shap_summary
from src.visualization.plot_metrics import plot_model_metrics

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def main():
    print("=== Wczytywanie danych ===")
    df = load_dataset()

    print("=== Przygotowanie cech ===")
    X_train, X_test, y_train, y_test = prepare_features(df)

    print("=== Trenowanie modeli ===")
    results, predictions = train_all_models(X_train, X_test, y_train, y_test)

    print("=== Ewaluacja modeli ===")
    metrics_df = compare_models(y_test, predictions)

    plot_model_metrics(
        metrics_df,
        save_path_rmse="results/plots/metrics_rmse.png",
        save_path_r2="results/plots/metrics_r2.png"
    )

    print("=== Analiza feature importance ===")
    feature_names = X_train.columns.tolist()

    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    xgb_model = XGBRegressor(random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)

    rf_imp = get_feature_importance_rf(rf_model, feature_names)
    xgb_imp = get_feature_importance_xgb(xgb_model, feature_names)

    plot_feature_importance_bar(
        importance_df=rf_imp,
        title="Random Forest - Feature Importance",
        save_path="results/plots/rf_importance.png"
    )

    plot_feature_importance_bar(
        importance_df=xgb_imp,
        title="XGBoost Feature Importance (Gain)",
        save_path="results/plots/xgb_importance_plot.png",
        metric="gain"
    )

    print("=== SHAP Summary Plot dla XGBoost ===")
    if hasattr(xgb_model, "get_booster"):
        plot_shap_summary(
            model=xgb_model,
            X_sample=X_test,
            feature_names=feature_names,
            save_path="results/plots/shap_summary_xgb.png"
        )
    else:
        print("SHAP nie działa dla tego modelu.")

if __name__ == "__main__":
    main()
