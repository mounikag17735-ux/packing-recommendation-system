import os
import pandas as pd
import joblib

from analytics.train_model import train_models
from analytics.cloud_db import get_connection


MODEL_PATH = "analytics/rf_cost_model.pkl"
ENCODER_PATH = "analytics/industry_encoder.pkl"


# ---------------------------------------------------
# Lazy load / auto-train models (HF compatible)
# ---------------------------------------------------
def get_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("🔧 Models missing. Training now...")
        train_models()

    rf_model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return rf_model, encoder


# ---------------------------------------------------
# Map UI values → DB values
# ---------------------------------------------------
INDUSTRY_MAP = {
    "Food": "Food & Beverage",
    "Pharmaceuticals": "Pharmaceutical",
    "Electronics": "Electronics",
    "Cosmetics": "Cosmetics",
}


# ---------------------------------------------------
# Load materials from SQLiteCloud
# ---------------------------------------------------
def load_materials():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM materials", conn)
    conn.close()
    return df


# ---------------------------------------------------
# Recommendation Logic
# ---------------------------------------------------
def recommend_material(fragility, weight, eco_priority, industry):
    rf_model, encoder = get_models()

    df = load_materials()

    # Clean + map input
    industry = industry.strip().title()
    db_industry = INDUSTRY_MAP.get(industry, industry)

    # Filter by industry
    df = df[df["industry_category"] == db_industry].copy()
    if df.empty:
        return []

    # Weight filter
    filtered = df[df["weight_capacity"] >= weight]
    if not filtered.empty:
        df = filtered

    # Encode industry like training
    encoded_industry = encoder.transform([db_industry])[0]
    df["industry_category"] = encoded_industry

    # Features EXACT as training
    features = df[
        [
            "strength",
            "weight_capacity",
            "biodegradability_score",
            "recyclability_percent",
            "industry_category",
        ]
    ].copy()

    # Predictions
    df["predicted_cost"] = rf_model.predict(features)

    max_co2 = df["co2_emission_score"].max()
    df["predicted_co2"] = df["co2_emission_score"] / max_co2

    # Smart score
    fragility_factor = fragility / 10
    df["fragility_score"] = df["strength"] * fragility_factor

    df["final_score"] = (
        df["predicted_cost"] * (1 - eco_priority)
        + (1 - df["predicted_co2"]) * eco_priority
        + df["fragility_score"] * 0.2
    )

    ranked = df.sort_values(by="final_score", ascending=False).head(5)

    # Save logs to Cloud DB
    conn = get_connection()
    cursor = conn.cursor()

    results = []

    for _, row in ranked.iterrows():
        explanation = f"""
Supports {weight}kg weight,
suitable for fragility level {fragility},
balances cost with eco priority {eco_priority}.
""".strip()

        cursor.execute(
            """
            INSERT INTO recommendation_logs
            (industry, material, predicted_cost, predicted_co2)
            VALUES (?, ?, ?, ?)
            """,
            (
                db_industry,
                row["material_type"],
                float(row["predicted_cost"]),
                float(row["predicted_co2"]),
            ),
        )

        results.append(
            {
                "material_name": row["material_type"],
                "predicted_cost": round(float(row["predicted_cost"]), 2),
                "predicted_co2": round(float(row["predicted_co2"]), 2),
                "explanation": explanation,
            }
        )

    conn.commit()
    conn.close()

    return results
