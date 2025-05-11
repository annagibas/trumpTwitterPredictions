import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_features(df):
    """
    Przygotowuje dane do modelowania:
    - tworzy nowe cechy tekstowe,
    - oddziela cechy i zmienną docelową,
    - dzieli dane na zbiory treningowe i testowe.

    Zwraca:
    - X_train, X_test, y_train, y_test
    """

    # 1. Walidacja obecności zmiennej docelowej
    if 'retweet_count' not in df.columns:
        raise ValueError("Brakuje kolumny 'retweet_count' w danych wejściowych")

    # 2. Tworzenie cech tekstowych
    df['length'] = df['text'].astype(str).apply(len)              # długość tweeta
    df['has_hashtag'] = df['text'].str.contains('#').astype(int)  # obecność hashtagu
    df['has_url'] = df['text'].str.contains('http').astype(int)   # obecność linku
    df['is_question'] = df['text'].astype(str).str.strip().str.endswith('?').astype(int)  # czy tweet kończy się znakiem zapytania

    # 3. Usunięcie kolumny tekstowej
    df = df.drop(columns=['text'])

    # 4. Oddzielenie zmiennej docelowej
    y = df['retweet_count']
    X = df.drop(columns=['retweet_count'])

    # 5. Nie ma potrzeby kodowania – wszystkie cechy są numeryczne/binarnie zakodowane
    X_final = X.copy()

    # 6. Podział na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test