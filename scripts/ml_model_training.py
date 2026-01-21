from pathlib import Path
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# =========================
# PATH SETUP (CRITICAL)
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# =========================
# LOAD ARTIFACTS
# =========================
X_train_processed = joblib.load(ARTIFACTS_DIR / "X_train.pkl")
X_test_processed  = joblib.load(ARTIFACTS_DIR / "X_test.pkl")

y_cost_train = joblib.load(ARTIFACTS_DIR / "y_cost_train.pkl")
y_cost_test  = joblib.load(ARTIFACTS_DIR / "y_cost_test.pkl")

y_co2_train = joblib.load(ARTIFACTS_DIR / "y_co2_train.pkl")
y_co2_test  = joblib.load(ARTIFACTS_DIR / "y_co2_test.pkl")

preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.pkl")
df = joblib.load(ARTIFACTS_DIR / "df.pkl")
X = joblib.load(ARTIFACTS_DIR / "X.pkl")

# =========================
# COST MODEL (RandomForest)
# =========================
cost_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

cost_model.fit(X_train_processed, y_cost_train)
y_cost_pred = cost_model.predict(X_test_processed)

# =========================
# CO2 MODEL (CPU-SAFE XGBOOST)
# =========================
co2_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",   # 🔥 CPU SAFE (NO GPU)
    predictor="cpu_predictor"
)

co2_model.fit(X_train_processed, y_co2_train)
y_co2_pred = co2_model.predict(X_test_processed)

# =========================
# EVALUATION
# =========================
def evaluate_model(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

evaluate_model(y_cost_test, y_cost_pred, "Cost Model")
evaluate_model(y_co2_test, y_co2_pred, "CO₂ Model")

# =========================
# SAVE MODELS (OVERWRITE)
# =========================
joblib.dump(cost_model, ARTIFACTS_DIR / "cost_model.pkl")
joblib.dump(co2_model, ARTIFACTS_DIR / "co2_model.pkl")

print("\n✅ Models trained and saved successfully")
