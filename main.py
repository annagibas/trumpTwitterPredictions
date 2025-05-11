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
from src.optuna_optimalization.optuna_optimalization import run_optuna_rf
from src.optuna_optimalization.optuna_optimalization import run_optuna_xgb
from src.utils.load_and_build_model import load_best_params, build_model_from_params
from src.utils.save_params import save_best_params
import os
from src.utils.eda_analysis import plot_target_distribution
from src.models.lasso_feature_selection import run_lasso
import pandas as pd

# Funkcja pomocnicza do zapisu Top 5 Feature Importance
def save_top5_feature_importance(rf_model, xgb_model, feature_names, output_dir="results/feature_importance"):
    """
    Zapisuje top 5 najważniejszych cech dla Random Forest i XGBoost do csv.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Random Forest Feature Importance
    rf_importances = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_model.feature_importances_
    })
    rf_top5 = rf_importances.sort_values(by="importance", ascending=False).head(5)
    rf_top5.to_csv(os.path.join(output_dir, "top5_rf_importance.csv"), index=False)

    # XGBoost Feature Importance (Gain)
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

    #Analiza rozkładu zmiennej docelowej
    print("Rozkład zmiennej docelowej")
    plot_target_distribution(df)

    print("Przygotowanie cech")
    X_train, X_test, y_train, y_test = prepare_features(df)

    #które cechy są najbardziej predykcyjne - tylko dla regresji liniowej
    print("Selekcja cech Lasso")
    run_lasso()

    # Optuna: szukanie najlepszych hiperparametrów dla Random Forest
    print("Optymalizacja Random Forest za pomocą Optuna")
    best_params_rf = run_optuna_rf(X_train, y_train, n_trials=50)

    # Optuna: szukanie najlepszych hiperparametrów dla xgb
    print("Optymalizacja XGBoost za pomocą Optuna")
    best_params_xgb = run_optuna_xgb(X_train, y_train, n_trials=50)

    # Zapis najlepszych hiperparametrów do plików
    save_best_params(best_params_rf, "results/params/best_params_rf.json")
    save_best_params(best_params_xgb, "results/params/best_params_xgb.json")

    # Wczytanie najlepszych hiperparametrów (opcjonalnie)
    best_params_rf = load_best_params("results/params/best_params_rf.json")
    best_params_xgb = load_best_params("results/params/best_params_xgb.json")

    # Budowanie modeli na podstawie załadowanych parametrów
    rf_model = build_model_from_params(RandomForestRegressor, best_params_rf)
    xgb_model = build_model_from_params(XGBRegressor, best_params_xgb)

    print("Trenowanie modeli")
    results, predictions, trained_models = train_all_models(X_train, X_test, y_train, y_test)

    print("Ewaluacja modeli")
    metrics_df = compare_models(y_test, predictions)

    plot_model_metrics(
        metrics_df,
        save_path_rmse="results/plots/metrics_rmse.png",
        save_path_r2="results/plots/metrics_r2.png"
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
    else:
        print("SHAP nie działa dla tego modelu.")

if __name__ == "__main__":
    main()
