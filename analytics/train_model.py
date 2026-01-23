import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ml_prepare import (
    load_data,
    engineer_targets,
    encode_features,
    select_features,
    split_data,
)


# ---------------------------------------------------
# Evaluation Function
# ---------------------------------------------------
def evaluate(model, X_test, y_test, title):
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n📊 {title}")
    print(f"RMSE: {rmse}")
    print(f"MAE : {mae}")
    print(f"R2  : {r2}")


# ---------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------
if __name__ == "__main__":
    print("Training ML models...")

    # Load and prepare data from Module 3
    df = load_data()
    df = engineer_targets(df)
    df = encode_features(df)

    X, y_cost, y_co2 = select_features(df)

    (
        X_train,
        X_test,
        y_cost_train,
        y_cost_test,
        y_co2_train,
        y_co2_test,
    ) = split_data(X, y_cost, y_co2)

    # ---------------------------------------------------
    # Model 1: Random Forest for Cost Prediction
    # ---------------------------------------------------
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_cost_train)

    evaluate(rf_model, X_test, y_cost_test, "Random Forest - Cost Prediction")

    # Save model
    joblib.dump(rf_model, "analytics/rf_cost_model.pkl")

    # ---------------------------------------------------
    # Model 2: XGBoost for CO2 Prediction
    # ---------------------------------------------------
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_co2_train)

    evaluate(xgb_model, X_test, y_co2_test, "XGBoost - CO2 Prediction")

    # Save model
    joblib.dump(xgb_model, "analytics/xgb_co2_model.pkl")

    print("\n✅ Models trained and saved successfully!")
