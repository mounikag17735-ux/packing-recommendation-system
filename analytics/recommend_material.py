import sqlite3
import pandas as pd
import joblib
import os
import shutil
from analytics.train_model import train_models

# Detect if running on Hugging Face (Linux container)
if os.name != "nt":
    DB_PATH = "/tmp/materials.db"
    if not os.path.exists(DB_PATH):
        shutil.copy("data/materials.db", DB_PATH)
else:
    # Local Windows run
    DB_PATH = "data/materials.db"



MODEL_PATH = "analytics/rf_cost_model.pkl"
ENCODER_PATH = "analytics/industry_encoder.pkl"

    # Auto-train models if not present (for HuggingFace)
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    train_models()

rf_model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


# 🔥 Map UI values → DB/Encoder values
INDUSTRY_MAP = {
    "Food": "Food & Beverage",
    "Pharmaceuticals": "Pharmaceutical",
    "Electronics": "Electronics",
    "Cosmetics": "Cosmetics",
}


def load_materials():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM materials", conn)
    conn.close()
    return df


def recommend_material(fragility, weight, eco_priority, industry):
    df = load_materials()

    # --- Clean input and map to DB value ---
    industry = industry.strip().title()
    db_industry = INDUSTRY_MAP.get(industry, industry)

    # --- Filter using DB value ---
    df = df[df["industry_category"] == db_industry].copy()
    if df.empty:
        return []

    # --- Weight filter with fallback ---
    filtered = df[df["weight_capacity"] >= weight]
    if not filtered.empty:
        df = filtered

    # --- Encode industry EXACTLY like training ---
    encoded_industry = encoder.transform([db_industry])[0]

    # --- Features EXACTLY same as training ---
    df["industry_category"] = encoded_industry

    features = df[
        [
            "strength",
            "weight_capacity",
            "biodegradability_score",
            "recyclability_percent",
            "industry_category",
        ]
    ].copy()

    # --- ML Predictions ---
    df["predicted_cost"] = rf_model.predict(features)

    max_co2 = df["co2_emission_score"].max()
    df["predicted_co2"] = df["co2_emission_score"] / max_co2

    # --- Smart scoring ---
    fragility_factor = fragility / 10
    df["fragility_score"] = df["strength"] * fragility_factor

    df["final_score"] = (
        df["predicted_cost"] * (1 - eco_priority)
        + (1 - df["predicted_co2"]) * eco_priority
        + df["fragility_score"] * 0.2
    )

    ranked = df.sort_values(by="final_score", ascending=False).head(5)

    results = []

    # --- Save logs for dashboard ---
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for _, row in ranked.iterrows():
        explanation = f"""
Supports {weight}kg weight,
suitable for fragility level {fragility},
balances cost with eco priority {eco_priority}.
""".strip()

        cursor.execute("""
            INSERT INTO recommendation_logs
            (industry, material, predicted_cost, predicted_co2)
            VALUES (?, ?, ?, ?)
        """, (
            db_industry,
            row["material_type"],
            float(row["predicted_cost"]),
            float(row["predicted_co2"])
        ))

        results.append({
            "material_name": row["material_type"],
            "predicted_cost": round(float(row["predicted_cost"]), 2),
            "predicted_co2": round(float(row["predicted_co2"]), 2),
            "explanation": explanation
        })

    conn.commit()
    conn.close()

    return results
