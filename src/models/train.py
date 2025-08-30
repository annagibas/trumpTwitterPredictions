from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np

def train_linear_regression(X_train, X_test, y_train, y_test):
    results = {}
    predictions = {}

    model = LinearRegression()
    y_train_log = np.log1p(y_train)
    model.fit(X_train, y_train_log)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    results = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'CV_RMSE_mean': -np.mean(cross_val_score(model, X_train, y_train_log, cv=5, scoring='neg_root_mean_squared_error'))
    }

    predictions = y_pred
    return results, predictions, model


def train_random_forest(X_train, X_test, y_train, y_test):
    results = {}
    predictions = {}

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'CV_RMSE_mean': -np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error'))
    }

    predictions = y_pred
    return results, predictions, model


def train_xgboost(X_train, X_test, y_train, y_test):
    results = {}
    predictions = {}

    model = XGBRegressor(random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'CV_RMSE_mean': -np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error'))
    }

    predictions = y_pred
    return results, predictions, model