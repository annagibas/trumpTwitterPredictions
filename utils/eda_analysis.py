import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno


def basic_info(df: pd.DataFrame):
    print("\nINFORMACJE O DATAFRAME\n")
    print(df.info())
    print("\nSTATYSTYKI OPISOWE\n")
    print(df.describe(include='all'))


def missing_data(df: pd.DataFrame):
    print("\nBRAKI DANYCH\n")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing values': missing, 'Percent': missing_percent})
    print(missing_df[missing_df['Missing values'] > 0])

    # Bezpieczna wizualizacja braków
    msno.bar(df)
    plt.title("Missing Data Bar Chart")
    plt.show()


def correlation_analysis(df: pd.DataFrame):
    print("\nKORELACJE MIĘDZY ZMIENNYMI NUMERYCZNYMI\n")
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Mapa korelacji")
    plt.show()


def detect_outliers_iqr(df: pd.DataFrame):
    print("\nDETEKCJA OUTLIERÓW (IQR)\n")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

        print(f"{col}: liczba wartości odstających: {len(outliers)}")

        plt.figure(figsize=(6, 1.5))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot dla {col}")
        plt.show()