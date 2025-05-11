import optuna
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import numpy as np

# Funkcja celu dla Random Forest
def objective_rf(trial, X_train, y_train):
    """
    Funkcja celu dla optymalizacji Random Forest za pomocą Optuna.
    Zwraca RMSE jako wartość do minimalizacji.
    Dodatkowo oblicza R2 dla celów informacyjnych (nieoptymalizowanych).
    """
    # Propozycje hiperparametrów
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    # Budowanie modelu z proponowanymi hiperparametrami
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    # Obliczanie R² na danych treningowych (nie wpływa na wynik optymalizacji)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    print(f"[INFO] R2 (RF, train set): {r2:.4f}")

    # Ocena modelu za pomocą cross-validation (główna metryka: RMSE)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    mean_score = np.mean(scores)

    return -mean_score  # wartość do minimalizacji


def run_optuna_rf(X_train, y_train, n_trials=50):
    """
    Funkcja uruchamiająca proces optymalizacji dla Random Forest.

    Parametry:
    X_train, y_train - dane treningowe
    n_trials - liczba prób (domyślnie 50)
    """
    study = optuna.create_study(
        direction='minimize',
        study_name='RF_Optimization',
        sampler=optuna.samplers.TPESampler(seed=42)  # dla powtarzalności
    )
    study.optimize(lambda trial: objective_rf(trial, X_train, y_train), n_trials=n_trials)

    print("Najlepsze parametry (Random Forest):")
    print(study.best_params)
    print(f"Najlepszy wynik (RMSE): {-study.best_value:.4f}")

    # Zapis prób (opcjonalnie)
    study.trials_dataframe().to_csv("results/optuna_rf_trials.csv", index=False)

    return study.best_params


# Funkcja celu dla XGBoost
def objective_xgb(trial, X_train, y_train):
    """
    Funkcja celu dla optymalizacji XGBoost za pomocą Optuna.
    Zwraca RMSE jako wartość do minimalizacji.
    Dodatkowo oblicza R2 dla celów interpretacyjnych.
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
    }

    model = XGBRegressor(
        **param,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )

    # Obliczanie R² na danych treningowych
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    print(f"[INFO] R2 (XGB, train set): {r2:.4f}")

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    mean_score = np.mean(scores)

    return -mean_score


def run_optuna_xgb(X_train, y_train, n_trials=50):
    """
    Funkcja uruchamiająca proces optymalizacji dla XGBoost.

    Parametry:
    X_train, y_train - dane treningowe
    n_trials - liczba prób (domyślnie 50)
    """
    study = optuna.create_study(
        direction='minimize',
        study_name='XGB_Optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=n_trials)

    print("Najlepsze parametry XGBoost:")
    print(study.best_params)
    print(f"Najlepszy wynik (RMSE): {-study.best_value:.4f}")

    # Zapis prób (opcjonalnie)
    study.trials_dataframe().to_csv("results/optuna_xgb_trials.csv", index=False)

    return study.best_params
