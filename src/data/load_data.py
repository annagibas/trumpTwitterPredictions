import pandas as pd
from pathlib import Path

def load_dataset():
    """
    Wczytywanie pliku Trump_data.csv z katalogu data/.
    Zwraca obiekt DataFrame z danymi.
    """

    data_path = Path(__file__).resolve().parents[2] / 'data' / 'Trump_data.csv'

    if not data_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {data_path}")

    df = pd.read_csv(data_path)

    print(f"Dane wczytane pomyślnie. Liczba rekordów: {df.shape[0]}, liczba kolumn: {df.shape[1]}")
    return df