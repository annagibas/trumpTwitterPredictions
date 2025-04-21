from pathlib import Path
import pandas as pd


def load_trump_data(file_name: str = "Trump_Data.csv") -> pd.DataFrame:
    # Obliczenie ścieżki do folderu głównego projektu
    project_root = Path(__file__).resolve().parents[1]

    # Ścieżka do pliku z danymi
    data_path = project_root / "Dataset" / file_name

    # Wczytanie danych z CSV
    df = pd.read_csv(data_path)

    print(f"[INFO] Dane wczytane z: {data_path}")
    print(f"[INFO] Rozmiar danych: {df.shape}")

    return df