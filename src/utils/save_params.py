import json
import os

def save_best_params(params: dict, save_path: str):
    """
    Zapisuje najlepsze hiperparametry do pliku .json.

    Args:
        params (dict): Słownik z hiperparametrami.
        save_path (str): Ścieżka do pliku .json.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Zapisano najlepsze parametry do {save_path}")