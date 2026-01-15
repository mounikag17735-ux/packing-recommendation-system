import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

DATA_PATH = "../data/materials_features.csv"
df = pd.read_csv(DATA_PATH)

numeric_features = [
    "STRENGTH",
    "WEIGHT_CAPACITY",
    "BIODEGRADABILITY_SCORE",
    "RECYCLABILITY_PERCENTAGE",
    "Co2_EMISSION_SCORE"
]

material_type_features = [
    col for col in df.columns if col.startswith("MATERIAL_TYPE_")
]

industry_features = [
    col for col in df.columns if col.startswith("INDUSTRY_CATEGORY_")
]

feature_columns = numeric_features + material_type_features + industry_features

X = df[feature_columns]

y_cost = df["Cost_Efficiency_Index"] 
y_co2 = df["CO2_Impact_Index"]          

X_train, X_test, y_cost_train, y_cost_test, y_co2_train, y_co2_test = train_test_split(
    X,
    y_cost,
    y_co2,
    test_size=0.2,
    random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("categorical", "passthrough", material_type_features + industry_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(" ML Dataset Preparation Completed Successfully!")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Processed training feature shape: {X_train_processed.shape}")
joblib.dump(X_train_processed, "../artifacts/X_train.pkl")
joblib.dump(X_test_processed, "../artifacts/X_test.pkl")
joblib.dump(y_cost_train, "../artifacts/y_cost_train.pkl")
joblib.dump(y_cost_test, "../artifacts/y_cost_test.pkl")
joblib.dump(y_co2_train, "../artifacts/y_co2_train.pkl")
joblib.dump(y_co2_test, "../artifacts/y_co2_test.pkl")
joblib.dump(preprocessor, "../artifacts/preprocessor.pkl")
joblib.dump(df, "../artifacts/df.pkl")
joblib.dump(X, "../artifacts/X.pkl")
