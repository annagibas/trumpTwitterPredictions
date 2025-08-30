import pandas as pd
from sklearn.model_selection import train_test_split

# Stała z nazwą zmiennej docelowej
TARGET_COLUMN = "retweet_count"

def prepare_features(df):
    """
    Przygotowuje dane do modelowania:
    - standaryzuje nazwy kolumn,
    - tworzy nowe cechy tekstowe,
    - usuwa cechy nieistotne,
    - oddziela cechy i zmienną docelową,
    - dzieli dane na zbiory treningowe i testowe.

    Zwraca:
    - X_train, X_test, y_train, y_test
    """

    # 0. Standaryzacja nazw kolumn do małych liter
    df.columns = df.columns.str.lower()

    # 1. Walidacja obecności zmiennej docelowej
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Brakuje kolumny '{TARGET_COLUMN}' w danych wejściowych")

    # 2. Tworzenie cech tekstowych
    df['length'] = df['text'].astype(str).apply(len)              # długość tweeta
    df['has_hashtag'] = df['text'].str.contains('#').astype(int)  # obecność hashtagu
    df['has_url'] = df['text'].str.contains('http').astype(int)   # obecność linku
    df['is_question'] = df['text'].astype(str).str.strip().str.endswith('?').astype(int)  # czy tweet kończy się znakiem zapytania

    # 3. Usunięcie kolumny tekstowej
    df = df.drop(columns=['text'])

    # 4. Usunięcie cechy silnie dominującej
    df = df.drop(columns=["favorite_count"], errors="ignore")

    # 5. Oddzielenie zmiennej docelowej
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # 6. Nie ma potrzeby kodowania – wszystkie cechy są numeryczne/binarnie zakodowane
    X_final = X.copy()

    # 7. Podział na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test