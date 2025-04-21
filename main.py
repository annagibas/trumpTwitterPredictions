from data_loader.load_data import load_trump_data
from data_loader.load_data import load_trump_data
from utils.eda_analysis import basic_info, missing_data, correlation_analysis, detect_outliers_iqr

if __name__ == "__main__":
    df = load_trump_data()
    print(df.head())

    if __name__ == "__main__":
        df = load_trump_data()

        basic_info(df)
        missing_data(df)
        correlation_analysis(df)
        detect_outliers_iqr(df)