import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def compare_models(y_true, predictions_dict):
    """
    Funkcja porównuje skuteczność modeli na podstawie ich predykcji.
    Wyświetla i zwraca DataFrame z RMSE i R² dla każdego modelu.

    Parametry:
    - y_true: rzeczywiste wartości (y_test)
    - predictions_dict: słownik z predykcjami każdego modelu
        np. {'Linear Regression': y_pred_lr, 'Random Forest': y_pred_rf, ...}
    """

    results = []

    for model_name, y_pred in predictions_dict.items():
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        results.append({
            'Model': model_name,
            'RMSE': round(rmse, 2),
            'R2': round(r2, 3)
        })

    results_df = pd.DataFrame(results).sort_values(by='RMSE')
    print("\nPorównanie modeli regresyjnych:\n")
    print(results_df.to_string(index=False))

    return results_df