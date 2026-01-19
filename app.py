from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
import sqlite3
import os

API_KEY = os.getenv("API_KEY", "packaging_ai_2026_secret")
DB_PATH = "data/packaging.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def log_recommendation(input_data, recommendations):
    conn = get_db_connection()
    cursor = conn.cursor()

    for rec in recommendations:
        cursor.execute("""
            INSERT INTO recommendation_logs (
                material_name,
                predicted_cost,
                predicted_co2,
                eco_priority,
                fragility_level,
                industry,
                product_weight
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            rec["material_name"],
            rec["predicted_cost"],
            rec["predicted_co2"],
            input_data.get("eco_priority"),
            input_data.get("fragility_level"),
            input_data.get("industry"),
            input_data.get("product_weight")
        ))

    conn.commit()
    conn.close()

def load_materials_from_db():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM materials", conn)
    conn.close()
    return df

def log_recommendations(recommendations_df):
    conn = get_db_connection()
    cursor = conn.cursor()

    for _, row in recommendations_df.iterrows():
        cursor.execute("""
            INSERT INTO recommendation_logs (
                material_name,
                predicted_cost,
                predicted_co2
            ) VALUES (?, ?, ?)
        """, (
            row.get("MATERIAL_TYPE"),
            row.get("Predicted_Cost"),
            row.get("Predicted_CO2")
        ))

    conn.commit()
    conn.close()

DEFAULT_MATERIAL = {
    "MATERIAL_TYPE": "Standard Packaging",
    "Predicted_Cost": None,
    "Predicted_CO2": None,
    "Final_Score": None,
    "Rank": 1,
    "Explanation": "Fallback recommendation due to insufficient matching data"
}

preprocessor = joblib.load("artifacts/preprocessor.pkl")
cost_model = joblib.load("artifacts/cost_model.pkl")
co2_model = joblib.load("artifacts/co2_model.pkl")
X = joblib.load("artifacts/X.pkl")

df = load_materials_from_db()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def ui():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "Packaging Recommendation API is running"
    })

@app.route("/recommend", methods=["POST"])
def recommend_material():
    api_key = request.headers.get("X-API-KEY")

    if api_key != API_KEY:
        return jsonify({
            "status": "error",
            "message": "Unauthorized"
        }), 401

    product_input = request.get_json()
    if not product_input:
        return jsonify({
            "status": "error",
            "message": "No input data provided"
        }), 400

    top_n = 5
    
    recommendations_df = generate_ai_recommendations(product_input, top_n)
    log_recommendations(recommendations_df)
    formatted_response = format_recommendation_response(recommendations_df)
    log_recommendation(product_input, formatted_response)

    def log_recommendations(input_product, recommended_materials):
        conn = get_db_connection()
        cursor = conn.cursor()

        for mat in recommended_materials:
            cursor.execute("""
                INSERT INTO recommendation_logs (
                    material_name, predicted_cost, predicted_co2,
                    eco_priority, fragility_level, industry, product_weight
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                mat["material_name"],
                mat["predicted_cost"],
                mat["predicted_co2"],
                input_product.get("eco_priority"),
                input_product.get("fragility_level"),
                input_product.get("industry"),
                input_product.get("product_weight")
            ))

        conn.commit()
        conn.close()
    log_recommendations(product_input, formatted_response)
    return jsonify({
        "status": "success",
        "input_product": product_input,
        "recommendation_count": top_n,
        "recommended_materials": formatted_response
    })

def generate_ai_recommendations(product_input, top_n=5):
    eco_priority = float(product_input.get("eco_priority", 0.5))
    fragility = product_input.get("fragility_level", "medium").strip().lower()
    product_weight = float(product_input.get("product_weight", 0))
    industry = product_input.get("industry", "electronics").strip().lower()

    # Fragility → strength threshold
    strength_map = {"low": 40, "medium": 60, "high": 80}
    min_strength = strength_map.get(fragility, 60)

    # Rule-based filtering
    filtered_df = df[
        (df["STRENGTH"] >= min_strength) &
        (df["WEIGHT_CAPACITY"] >= product_weight) &
        (df["INDUSTRY_CATEGORY"].str.lower() == industry)
    ].copy()

    if filtered_df.empty:
        return pd.DataFrame([DEFAULT_MATERIAL])

    # ML feature prep
    X_filtered = X.loc[filtered_df.index]
    X_processed = preprocessor.transform(X_filtered)

    # Predictions
    filtered_df["Predicted_Cost"] = cost_model.predict(X_processed)
    filtered_df["Predicted_CO2"] = co2_model.predict(X_processed)

    # Normalization (safe)
    cost_min, cost_max = filtered_df["Predicted_Cost"].min(), filtered_df["Predicted_Cost"].max()
    cost_range = cost_max - cost_min

    co2_min, co2_max = filtered_df["Predicted_CO2"].min(), filtered_df["Predicted_CO2"].max()
    co2_range = co2_max - co2_min

    filtered_df["Cost_Score"] = 0 if cost_range == 0 else (
        (filtered_df["Predicted_Cost"] - cost_min) / cost_range
    )

    filtered_df["CO2_Score"] = 0 if co2_range == 0 else (
        (filtered_df["Predicted_CO2"] - co2_min) / co2_range
    )

    # Final AI score
    filtered_df["Final_Score"] = (
        (1 - eco_priority) * filtered_df["Cost_Score"] +
        eco_priority * filtered_df["CO2_Score"]
    )

    # Ranking
    filtered_df["Rank"] = filtered_df["Final_Score"].rank(ascending=True)

    # Explanation
    median_cost = filtered_df["Predicted_Cost"].median()
    median_co2 = filtered_df["Predicted_CO2"].median()

    filtered_df["Explanation"] = filtered_df.apply(
        lambda row: generate_explanation(row, eco_priority, median_cost, median_co2),
        axis=1
    )

    return filtered_df.sort_values("Rank").head(top_n).reset_index(drop=True)

def generate_explanation(row, eco_priority, median_cost, median_co2):
    reasons = []

    if row["Predicted_Cost"] <= median_cost:
        reasons.append("low predicted cost")

    if row["Predicted_CO2"] <= median_co2:
        reasons.append("low carbon footprint")

    if eco_priority > 0.7:
        reasons.append("aligned with high eco priority")

    return "Recommended due to " + ", ".join(reasons)

def get_baseline_metrics():
    conn = get_db_connection()

    df_all = pd.read_sql("""
        SELECT AVG(Co2_EMISSION_SCORE) AS avg_co2,
               AVG(Cost_Efficiency_Index) AS avg_cost
        FROM materials
    """, conn)

    conn.close()

    return {
        "baseline_co2": df_all["avg_co2"].iloc[0],
        "baseline_cost": df_all["avg_cost"].iloc[0]
    }

def compute_sustainability_metrics(predicted_cost, predicted_co2):
    baseline = get_baseline_metrics()

    co2_reduction_pct = (
        (baseline["baseline_co2"] - predicted_co2)
        / baseline["baseline_co2"]
    ) * 100

    cost_savings_pct = (
        (baseline["baseline_cost"] - predicted_cost)
        / baseline["baseline_cost"]
    ) * 100

    return {
        "co2_reduction_percent": round(co2_reduction_pct, 2),
        "cost_savings_percent": round(cost_savings_pct, 2)
    }

def format_recommendation_response(df):
    formatted = []

    for _, row in df.iterrows():
        formatted.append({
            "material_name": row.get("MATERIAL_TYPE", "Standard Packaging"),
            "rank": int(row.get("Rank", 1)),
            "predicted_cost": round(row["Predicted_Cost"], 2)
                if pd.notna(row.get("Predicted_Cost")) else None,
            "predicted_co2": round(row["Predicted_CO2"], 2)
                if pd.notna(row.get("Predicted_CO2")) else None,
            "final_score": round(row["Final_Score"], 4)
                if pd.notna(row.get("Final_Score")) else None,
            "explanation": row.get("Explanation")
        })

    return formatted

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
