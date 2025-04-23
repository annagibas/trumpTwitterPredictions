from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def train_all_models(X_train, X_test, y_train, y_test):
    """
    Funkcja trenuje trzy modele regresyjne:
    - regresję liniową (ze standaryzacją),
    - Random Forest (bez standaryzacji),
    - XGBoost (bez standaryzacji).

    Zwraca:
    - słownik z wynikami metryk (RMSE, R2),
    - słownik z predykcjami każdego modelu.
    """

    results = {}
    predictions = {}

    # 1. Regresja liniowa (ze standaryzacją)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    results['Linear Regression'] = {
        'RMSE': mean_squared_error(y_test, y_pred_lr, squared=False),
        'R2': r2_score(y_test, y_pred_lr)
    }
    predictions['Linear Regression'] = y_pred_lr

    # 2. Random Forest (bez standaryzacji)
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    results['Random Forest'] = {
        'RMSE': mean_squared_error(y_test, y_pred_rf, squared=False),
        'R2': r2_score(y_test, y_pred_rf)
    }
    predictions['Random Forest'] = y_pred_rf

    # 3. XGBoost (bez standaryzacji)
    xgb_model = XGBRegressor(random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    results['XGBoost'] = {
        'RMSE': mean_squared_error(y_test, y_pred_xgb, squared=False),
        'R2': r2_score(y_test, y_pred_xgb)
    }
    predictions['XGBoost'] = y_pred_xgb

    return results, predictions