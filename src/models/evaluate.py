import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def compare_models(y_true, predictions_dict):
    """
    Funkcja porównuje skuteczność modeli na podstawie ich predykcji.
    Wyświetla i zwraca DataFrame z MAE, RMSE i R² dla każdego modelu.

    Parametry:
    - y_true: rzeczywiste wartości (y_test)
    - predictions_dict: słownik z predykcjami każdego modelu
        np. {'Linear Regression': y_pred_lr, 'Random Forest': y_pred_rf, ...}
    """

    if not isinstance(predictions_dict, dict):
        raise ValueError("Predictions_dict musi być słownikiem: {model_name: y_pred}")

    results = []

    for model_name, y_pred in predictions_dict.items():
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        results.append({
            'Model': model_name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 3)
        })

    results_df = pd.DataFrame(results).sort_values(by='RMSE')
    print("\nPorównanie modeli regresyjnych:\n")
    print(results_df.to_string(index=False))
    return results_df