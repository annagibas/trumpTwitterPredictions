from src.data.load_data import load_dataset
from src.utils.feature_engineering import prepare_features
from src.models.train import train_linear_regression, train_random_forest, train_xgboost
from src.models.evaluate import compare_models
from src.models.feature_importance import get_feature_importance_rf, get_feature_importance_xgb
from src.visualization.plot_feature_importance import plot_feature_importance_bar
from src.visualization.plot_shap_summary import plot_shap_summary
from src.visualization.plot_metrics import plot_model_metrics
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.optuna_optimalization.optuna_optimalization import run_optuna_rf, run_optuna_xgb
from src.utils.load_and_build_model import load_best_params
from src.utils.save_params import save_best_params
from src.utils.eda_analysis import plot_target_distribution
from src.models.lasso_feature_selection import run_lasso
import os
import pandas as pd
import shap


def save_top5_feature_importance(rf_model, xgb_model, feature_names, output_dir="results/feature_importance"):
    """
    Zapisuje top 5 najważniejszych cech dla Random Forest i XGBoost do csv.
    """
    os.makedirs(output_dir, exist_ok=True)

    rf_importances = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_model.feature_importances_
    })
    rf_top5 = rf_importances.sort_values(by="importance", ascending=False).head(5)
    rf_top5.to_csv(os.path.join(output_dir, "top5_rf_importance.csv"), index=False)

    xgb_booster = xgb_model.get_booster()
    xgb_importances_dict = xgb_booster.get_score(importance_type="gain")
    xgb_importances = pd.DataFrame({
        "Feature": list(xgb_importances_dict.keys()),
        "Importance": list(xgb_importances_dict.values())
    })
    xgb_top5 = xgb_importances.sort_values(by="Importance", ascending=False).head(5)
    xgb_top5.to_csv(os.path.join(output_dir, "top5_xgb_importance.csv"), index=False)

    print("Top 5 cech zapisane do katalogu:", output_dir)


def main():
    print("Wczytywanie danych")
    df = load_dataset()

    print("Rozkład zmiennej docelowej")
    plot_target_distribution(df)

    print("Przygotowanie cech")
    X_train, X_test, y_train, y_test = prepare_features(df)

    print("Selekcja cech Lasso")
    run_lasso()

    print("Optymalizacja Random Forest za pomocą Optuna")
    best_params_rf = run_optuna_rf(X_train, y_train, n_trials=5)

    print("Optymalizacja XGBoost za pomocą Optuna")
    best_params_xgb = run_optuna_xgb(X_train, y_train, n_trials=5)

    save_best_params(best_params_rf, "results/params/best_params_rf.json")
    save_best_params(best_params_xgb, "results/params/best_params_xgb.json")

    best_params_rf = load_best_params("results/params/best_params_rf.json")
    best_params_xgb = load_best_params("results/params/best_params_xgb.json")

    rf_model = RandomForestRegressor(**best_params_rf)
    xgb_model = XGBRegressor(**best_params_xgb)

    print("Trenowanie modeli")
    results = {}
    predictions = {}
    trained_models = {}

    results["LinearRegression"], predictions["LinearRegression"], trained_models["LinearRegression"] = train_linear_regression(X_train, X_test, y_train, y_test)
    results["RandomForest"], predictions["RandomForest"], trained_models["RandomForest"] = train_random_forest(X_train, X_test, y_train, y_test)
    results["XGBoost"], predictions["XGBoost"], trained_models["XGBoost"] = train_xgboost(X_train, X_test, y_train, y_test)

    results_df = pd.DataFrame(results).T
    os.makedirs("results/metrics", exist_ok=True)
    results_df.to_csv("results/metrics/model_metrics.csv")
    print("Wyniki metryk zapisane do: results/metrics/model_metrics.csv")

    print("Ewaluacja modeli")
    metrics_df = compare_models(y_test, predictions)

    plot_model_metrics(
        metrics_df,
        save_path_rmse="results/plots/metrics_rmse.png",
        save_path_r2="results/plots/metrics_r2.png",
        save_path_mae="results/plots/metrics_mae.png"
    )

    print("Analiza feature importance")
    feature_names = X_train.columns.tolist()

    rf_model = trained_models['RandomForest']
    xgb_model = trained_models['XGBoost']

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
        metric="Importance"
    )

    print("SHAP Summary Plot dla XGBoost")
    if hasattr(xgb_model, "get_booster"):
        plot_shap_summary(
            model=xgb_model,
            X_sample=X_test,
            feature_names=feature_names,
            save_path="results/plots/shap_summary_xgb.png"
        )

        # Analiza SHAP tylko dla cechy "followers"
        print("\nAnaliza SHAP dla cechy 'followers':")
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(X_test)

        if 'followers' in X_test.columns:
            idx = X_test.columns.get_loc('followers')
            followers_shap = shap_values.values[:, idx]

            print(f"Średnia wartość SHAP dla 'followers': {followers_shap.mean():.2f}")
            print(f"Zakres wartości SHAP dla 'followers': od {followers_shap.min():.2f} do {followers_shap.max():.2f}")
            print("Pozytywne wartości SHAP → zwiększają liczbę retweetów, negatywne → zmniejszają.")
        else:
            print("Cechy 'followers' brak w zbiorze testowym.")
    else:
        print("SHAP nie działa dla tego modelu.")


if __name__ == "__main__":
    main()