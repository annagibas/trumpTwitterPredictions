from sklearn.model_selection import train_test_split

def prepare_features(df):
    """
    Przygotowuje dane do modelowania:
    - oddziela cechy i zmienną docelową,
    - dzieli dane na zbiory treningowe i testowe.

    Zakłada, że kolumna 'retweets' jest zmienną docelową.

    Zwraca:
    - X_train, X_test, y_train, y_test
    """

    # Upewnij się, że kolumna 'retweets' istnieje
    if 'retweet_count' not in df.columns:
        raise ValueError("Brakuje kolumny 'retweet_count' w danych wejściowych")

    # Oddziel zmienną docelową
    X = df.drop(columns=['retweet_count'])
    y = df['retweet_count']

    #usuwanie tesktu, bo niepotrzbny
    X = df.drop(columns=["retweet_count", "text"])
    y = df["retweet_count"]

    # Podział danych na treningowe i testowe (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test