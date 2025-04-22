from data_loader.load_data import load_trump_data
from utils.eda_analysis import basic_info, missing_data, correlation_analysis, detect_outliers_iqr
from data_loader.load_data import load_trump_data
from utils.feature_engineering import prepare_features

#eda
if __name__ == "__main__":
    df = load_trump_data()
    print(df.head())

    basic_info(df)
    missing_data(df)
    correlation_analysis(df)
    detect_outliers_iqr(df)

#feature_engeniering
    if __name__ == "__main__":
        df = load_trump_data()
        df_prepared = prepare_features(df)
        print(df_prepared.head())