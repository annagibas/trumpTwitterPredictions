from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.data.load_data import load_dataset
from src.utils.feature_engineering import prepare_features

def run_lasso():
    # Wczytanie i przygotowanie danych
    df = load_dataset()
    X_train, X_test, y_train, y_test = prepare_features(df)

    # Standaryzacja cech
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Trening modelu Lasso z doborem alpha
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_train_scaled, y_train)

    # Współczynniki cech
    coef = pd.Series(lasso.coef_, index=X_train.columns)
    print("\nWspółczynniki Lasso (posortowane):")
    print(coef.sort_values(ascending=False))

    # Wybrane cechy
    selected_features = coef[coef != 0].index.tolist()
    print("\nWybrane cechy przez Lasso:")
    print(selected_features)

if __name__ == "__main__":
    run_lasso()