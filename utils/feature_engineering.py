import pandas as pd
import numpy as np


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Nowe cechy czasowe
    df['is_weekend'] = df['Week'] % 7 >= 5
    df['is_night'] = df['Hour'].apply(lambda x: 0 <= x <= 6)

    # Transformacje logarytmiczne (dodajemy 1, by uniknąć log(0))
    df['log_retweet_count'] = np.log1p(df['retweet_count'])
    df['log_favorite_count'] = np.log1p(df['favorite_count'])
    df['log_follower_change'] = np.log1p(df['Follower_Change'].abs())
    df['log_num_tweets'] = np.log1p(df['Num_Tweets'])

    # Wybór cech do modelowania
    features = df[[
        'log_retweet_count',
        'log_favorite_count',
        'log_follower_change',
        'log_num_tweets',
        'Followers',
        'Year',
        'Hour',
        'is_weekend',
        'is_night'
    ]]

    # Zamiana wartości logicznych na liczby
    features['is_weekend'] = features['is_weekend'].astype(int)
    features['is_night'] = features['is_night'].astype(int)

    return features