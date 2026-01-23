import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

from analytics.cloud_db import get_connection


# ---------------------------------------------------
# STEP 1: Load data from SQLite CLOUD
# ---------------------------------------------------
def load_data():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM materials", conn)
    conn.close()
    return df


# ---------------------------------------------------
# STEP 2: Create Target Columns
# ---------------------------------------------------
def engineer_targets(df):
    df["cost_efficiency_index"] = (
        df["strength"] + df["recyclability_percent"]
    ) / 2

    df["co2_impact_index"] = 1 - df["co2_emission_score"]

    return df


# ---------------------------------------------------
# STEP 3: Encode Categorical Columns
# ---------------------------------------------------
def encode_features(df):
    le = LabelEncoder()
    df["industry_category"] = le.fit_transform(df["industry_category"])

    # Save encoder for recommendation stage
    joblib.dump(le, "analytics/industry_encoder.pkl")

    return df


# ---------------------------------------------------
# STEP 4: Select ML Features and Targets
# ---------------------------------------------------
def select_features(df):
    X = df[
        [
            "strength",
            "weight_capacity",
            "biodegradability_score",
            "recyclability_percent",
            "industry_category",
        ]
    ]

    y_cost = df["cost_efficiency_index"]
    y_co2 = df["co2_emission_score"]

    return X, y_cost, y_co2


# ---------------------------------------------------
# STEP 5: Train/Test Split
# ---------------------------------------------------
def split_data(X, y_cost, y_co2):
    X_train, X_test, y_cost_train, y_cost_test = train_test_split(
        X, y_cost, test_size=0.2, random_state=42
    )

    _, _, y_co2_train, y_co2_test = train_test_split(
        X, y_co2, test_size=0.2, random_state=42
    )

    return (
        X_train,
        X_test,
        y_cost_train,
        y_cost_test,
        y_co2_train,
        y_co2_test,
    )
