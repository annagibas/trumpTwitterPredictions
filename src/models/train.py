from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np

def train_all_models(X_train, X_test, y_train, y_test):
    """
    Trenuje trzy modele regresyjne (regresja liniowa, Random Forest, XGBoost),
    dokonuje predykcji oraz ocenia jakość modeli.

    Dodatkowo:
    - wykonuje log-transformację zmiennej docelowej dla regresji liniowej,
    - oblicza dodatkowe metryki (MAE),
    - przeprowadza ocenę przez cross-validation,
    - zwraca także wytrenowane modele.
    """

    results = {}
    predictions = {}
    models = {}

    # REGRESJA LINIOWA
    lr = LinearRegression()

    # Log-transformacja y do regresji liniowej
    y_train_log = np.log1p(y_train)
    lr.fit(X_train, y_train_log)
    y_pred_log = lr.predict(X_test)
    y_pred_lr = np.expm1(y_pred_log)

    results['LinearRegression'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'R2': r2_score(y_test, y_pred_lr),
        'MAE': mean_absolute_error(y_test, y_pred_lr)
    }
    predictions['LinearRegression'] = y_pred_lr
    models['LinearRegression'] = lr

    # Cross-validation (RMSE) dla regresji liniowej
    cv_scores_lr = cross_val_score(lr, X_train, y_train_log, cv=5, scoring='neg_root_mean_squared_error')
    results['LinearRegression']['CV_RMSE_mean'] = -np.mean(cv_scores_lr)

    # RANDOM FOREST
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    results['RandomForest'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'R2': r2_score(y_test, y_pred_rf),
        'MAE': mean_absolute_error(y_test, y_pred_rf)
    }
    predictions['RandomForest'] = y_pred_rf
    models['RandomForest'] = rf

    # Cross-validation (RMSE) dla RF
    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    results['RandomForest']['CV_RMSE_mean'] = -np.mean(cv_scores_rf)

    # XGBOOST
    xgb = XGBRegressor(random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    results['XGBoost'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        'R2': r2_score(y_test, y_pred_xgb),
        'MAE': mean_absolute_error(y_test, y_pred_xgb)
    }
    predictions['XGBoost'] = y_pred_xgb
    models['XGBoost'] = xgb

    # Cross-validation (RMSE) dla XGBoost
    cv_scores_xgb = cross_val_score(xgb, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    results['XGBoost']['CV_RMSE_mean'] = -np.mean(cv_scores_xgb)

    return results, predictions, models