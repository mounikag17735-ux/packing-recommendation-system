import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

X_train_processed = joblib.load("../artifacts/X_train.pkl")
X_test_processed = joblib.load("../artifacts/X_test.pkl")

y_cost_train = joblib.load("../artifacts/y_cost_train.pkl")
y_cost_test = joblib.load("../artifacts/y_cost_test.pkl")

y_co2_train = joblib.load("../artifacts/y_co2_train.pkl")
y_co2_test = joblib.load("../artifacts/y_co2_test.pkl")

preprocessor = joblib.load("../artifacts/preprocessor.pkl")
df = joblib.load("../artifacts/df.pkl")   # original materials dataframe
X = joblib.load("../artifacts/X.pkl")     # full feature matrix

cost_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

cost_model.fit(X_train_processed, y_cost_train)
y_cost_pred = cost_model.predict(X_test_processed)

co2_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

co2_model.fit(X_train_processed, y_co2_train)
y_co2_pred = co2_model.predict(X_test_processed)

def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Performance")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

    return rmse, mae, r2

# Evaluate both models
evaluate_model(y_cost_test, y_cost_pred, "Cost Prediction (Random Forest)")
evaluate_model(y_co2_test, y_co2_pred, "CO₂ Impact Prediction (XGBoost)")

X_all_processed = preprocessor.transform(X)

cost_predictions_all = cost_model.predict(X_all_processed)
co2_predictions_all = co2_model.predict(X_all_processed)

ranking_df = df[["MATERIAL_ID", "MATERIAL_TYPE"]].copy()

ranking_df["Predicted_Cost"] = cost_predictions_all
ranking_df["Predicted_CO2"] = co2_predictions_all

# Normalize scores (lower is better)
ranking_df["Cost_Score"] = (
    ranking_df["Predicted_Cost"] - ranking_df["Predicted_Cost"].min()
) / (
    ranking_df["Predicted_Cost"].max() - ranking_df["Predicted_Cost"].min()
)

ranking_df["CO2_Score"] = (
    ranking_df["Predicted_CO2"] - ranking_df["Predicted_CO2"].min()
) / (
    ranking_df["Predicted_CO2"].max() - ranking_df["Predicted_CO2"].min()
)

eco_priority = 0.5  # default value (UI will override this)

ranking_df["Final_Recommendation_Score"] = (
    (1 - eco_priority) * ranking_df["Cost_Score"] +
    eco_priority * ranking_df["CO2_Score"]
)

ranking_df["Rank"] = ranking_df["Final_Recommendation_Score"].rank(
    ascending=True, method="dense"
)

top_materials = ranking_df.sort_values("Rank").head(10)

print("\nTop 10 Recommended Materials")
print(top_materials)

joblib.dump(cost_model, "../artifacts/cost_model.pkl")
joblib.dump(co2_model, "../artifacts/co2_model.pkl")

print("\nTrained models saved successfully!")
