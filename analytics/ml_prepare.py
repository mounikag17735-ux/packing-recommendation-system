import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

DB_PATH = "data/materials.db"


# ---------------------------------------------------
# STEP 1: Load data from SQLite
# ---------------------------------------------------
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM materials", conn)
    conn.close()
    return df


# ---------------------------------------------------
# STEP 2: Create Target Columns (if not already present)
# ---------------------------------------------------
def engineer_targets(df):
    # Cost Efficiency = strength + recyclability
    df["cost_efficiency_index"] = (
        df["strength"] + df["recyclability_percent"]
    ) / 2

    # CO2 Impact Index (lower emission = higher score)
    df["co2_impact_index"] = 1 - df["co2_emission_score"]

    return df


# ---------------------------------------------------
# STEP 3: Encode Categorical Columns
# ---------------------------------------------------
def encode_features(df):
    le = LabelEncoder()
    df["industry_category"] = le.fit_transform(df["industry_category"])

    # Save encoder for later use
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
    y_co2 = df["co2_emission_score"]  # ✅ real target

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


# ---------------------------------------------------
# MAIN PIPELINE (for testing Module 3)
# ---------------------------------------------------
if __name__ == "__main__":
    print("Preparing ML dataset...")

    df = load_data()
    df = engineer_targets(df)
    df = encode_features(df)

    X, y_cost, y_co2 = select_features(df)

    data = split_data(X, y_cost, y_co2)

    print("✅ ML dataset ready for training")
