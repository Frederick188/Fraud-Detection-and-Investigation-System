import pandas as pd
from sklearn.preprocessing import StandardScaler
import requests
import os
from dotenv import load_dotenv

load_dotenv()

IPGEO_API_KEY = os.getenv("IPGEO_API_KEY")
CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY")


def preprocess(df):
    """
    Preprocess the dataset.

    Steps:
    1. Detect target column.
    2. Keep only numeric features.
    3. Remove duplicate amount feature.
    4. Fill missing values.
    5. Standardize numeric features.

    Returns:
        X : Preprocessed feature dataframe
        y : Target series
        scaler : Fitted StandardScaler
    """

    df = df.copy()

    # Detect target column
    target_col = None
    for col in df.columns:
        col_name = col.lower().replace(" ", "_")
        if any(x in col_name for x in
               ["fraud", "fraud_flag", "fraud_flag_or_label",
                "class", "is_fraud"]):
            target_col = col
            break

    if target_col is None:
        raise ValueError(
            "No target column found "
            "(expected Fraud Flag, Class or Is_Fraud)"
        )

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # Keep only numeric columns
    X = X.select_dtypes(include=["number"])

    # Remove duplicate amount feature
    if "normalized_amount" in X.columns and "Transaction Amount" in X.columns:
        X = X.drop(columns=["Transaction Amount"])

    # Fill missing values
    X = X.fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=df.index
    )

    # Debug
    print("\nFeatures used for training:")
    print(X.columns.tolist())

    return X, y, scaler


def enrich_dataset(df):
    """
    Enrich dataset using CurrencyFreaks and IPGeolocation APIs.
    """

    enriched_rows = []

    for _, row in df.iterrows():
        new_row = row.copy()

        # ---------------- Currency Normalization ---------------- #
        try:
            currency = row.get("Transaction Currency", "USD")
            amount = float(row.get("Transaction Amount", 0))

            url = (
                f"https://api.currencyfreaks.com/latest?"
                f"apikey={CURRENCY_API_KEY}&symbols=USD,{currency}"
            )

            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                rates = response.json().get("rates", {})
                rate = float(rates.get(currency, 1))
                new_row["normalized_amount"] = round(amount / rate, 2)
            else:
                new_row["normalized_amount"] = amount

        except Exception:
            new_row["normalized_amount"] = amount

        # ---------------- IP Geolocation ---------------- #
        ip = row.get("IP Address")

        try:
            if ip:
                geo_url = (
                    f"https://api.ipgeolocation.io/ipgeo"
                    f"?apiKey={IPGEO_API_KEY}&ip={ip}"
                )

                geo = requests.get(geo_url, timeout=5)

                if geo.status_code == 200:
                    data = geo.json()

                    new_row["city"] = data.get("city", "Unknown")
                    new_row["country"] = data.get("country_code2", "Unknown")
                    new_row["latitude"] = float(data.get("latitude", 0))
                    new_row["longitude"] = float(data.get("longitude", 0))

                else:
                    new_row["city"] = "Unknown"
                    new_row["country"] = "Unknown"
                    new_row["latitude"] = 0
                    new_row["longitude"] = 0

            else:
                new_row["city"] = "Unknown"
                new_row["country"] = "Unknown"
                new_row["latitude"] = 0
                new_row["longitude"] = 0

        except Exception:
            new_row["city"] = "Unknown"
            new_row["country"] = "Unknown"
            new_row["latitude"] = 0
            new_row["longitude"] = 0

        enriched_rows.append(new_row)

    enriched_df = pd.DataFrame(enriched_rows)

    return enriched_df