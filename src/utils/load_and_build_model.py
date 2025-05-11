import json


def load_best_params(filepath):
    """
    Wczytuje hiperparametry zapisane w pliku .json.

    Args:
        filepath (str): Ścieżka do pliku .json.

    Returns:
        dict: Załadowane hiperparametry.
    """
    with open(filepath, 'r') as f:
        params = json.load(f)
    return params

def build_model_from_params(model_class, params):
    """
    Tworzy model na podstawie załadowanych hiperparametrów.

    Args:
        model_class (klasa): np. RandomForestRegressor lub XGBRegressor
        params (dict): Słownik z hiperparametrami.

    Returns:
        model: Gotowy model z ustawionymi hiperparametrami.
    """
    model = model_class(**params)
    return model