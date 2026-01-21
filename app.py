from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
import sqlite3
import os
from analytics.bi_metrics import get_bi_metrics
from analytics.bi_charts import generate_charts
from flask import send_file
from bi_dashboard.export_reports import load_logs, export_excel_report
from bi_dashboard.generate_pdf_report import load_logs as load_logs_pdf
from bi_dashboard.generate_pdf_report import generate_pdf
# -------------------- CONFIG --------------------
API_KEY = os.getenv("API_KEY", "packaging_ai_2026_secret")
DB_PATH = "data/packaging.db"

# -------------------- DB HELPERS --------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def load_materials_from_db():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM materials", conn)
    conn.close()
    return df

def log_recommendation(input_product, recommended_materials):
    conn = get_db_connection()
    cursor = conn.cursor()

    for mat in recommended_materials:
        cursor.execute("""
            INSERT INTO recommendation_logs (
                material_name,
                predicted_cost,
                predicted_co2,
                eco_priority,
                fragility_level,
                industry,
                product_weight
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

# -------------------- ML ARTIFACTS --------------------
preprocessor = joblib.load("artifacts/preprocessor.pkl")
cost_model = joblib.load("artifacts/cost_model.pkl")
co2_model = joblib.load("artifacts/co2_model.pkl")
X = joblib.load("artifacts/X.pkl")

df = load_materials_from_db()

DEFAULT_MATERIAL = {
    "MATERIAL_TYPE": "Standard Packaging",
    "Predicted_Cost": None,
    "Predicted_CO2": None,
    "Final_Score": None,
    "Rank": 1,
    "Explanation": "Fallback recommendation due to insufficient matching data"
}

# -------------------- FLASK APP --------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def ui():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    # Generate charts (saved as PNGs)
    generate_charts()

    # Get BI metrics
    metrics = get_bi_metrics()

    return render_template(
        "dashboard.html",
        metrics=metrics
    )

@app.route("/export/excel")
def export_excel():
    df = load_logs()

    if df.empty:
        return jsonify({"error": "No data available"}), 400

    export_excel_report(df)

    latest_file = sorted(os.listdir("reports"))[-1]
    return send_file(
        f"reports/{latest_file}",
        as_attachment=True
    )

@app.route("/export/pdf")
def export_pdf():
    path = generate_pdf()
    return send_file(path, as_attachment=True)


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
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    product_input = request.get_json()
    if not product_input:
        return jsonify({"status": "error", "message": "No input data provided"}), 400

    top_n = 5

    recommendations_df = generate_ai_recommendations(product_input, top_n)
    formatted_response = format_recommendation_response(recommendations_df)

    # Log once (correct)
    log_recommendation(product_input, formatted_response)

    return jsonify({
        "status": "success",
        "input_product": product_input,
        "recommendation_count": top_n,
        "recommended_materials": formatted_response
    })

# -------------------- AI LOGIC --------------------
def generate_ai_recommendations(product_input, top_n=5):
    eco_priority = float(product_input.get("eco_priority", 0.5))
    fragility = product_input.get("fragility_level", "medium").lower()
    product_weight = float(product_input.get("product_weight", 0))
    industry = product_input.get("industry", "electronics").lower()

    strength_map = {"low": 40, "medium": 60, "high": 80}
    min_strength = strength_map.get(fragility, 60)

    filtered_df = df[
        (df["STRENGTH"] >= min_strength) &
        (df["WEIGHT_CAPACITY"] >= product_weight) &
        (df["INDUSTRY_CATEGORY"]
            .str.lower()
            .str.contains(industry, na=False))
    ].copy()

    if filtered_df.empty:
        return pd.DataFrame([DEFAULT_MATERIAL])

    X_filtered = X.loc[filtered_df.index]
    X_processed = preprocessor.transform(X_filtered)

    filtered_df["Predicted_Cost"] = cost_model.predict(X_processed)
    filtered_df["Predicted_CO2"] = co2_model.predict(X_processed)

    cost_range = filtered_df["Predicted_Cost"].max() - filtered_df["Predicted_Cost"].min()
    co2_range = filtered_df["Predicted_CO2"].max() - filtered_df["Predicted_CO2"].min()

    filtered_df["Cost_Score"] = 0 if cost_range == 0 else (
        (filtered_df["Predicted_Cost"] - filtered_df["Predicted_Cost"].min()) / cost_range
    )

    filtered_df["CO2_Score"] = 0 if co2_range == 0 else (
        (filtered_df["Predicted_CO2"] - filtered_df["Predicted_CO2"].min()) / co2_range
    )

    filtered_df["Final_Score"] = (
        (1 - eco_priority) * filtered_df["Cost_Score"] +
        eco_priority * filtered_df["CO2_Score"]
    )

    filtered_df["Rank"] = (
        filtered_df["Final_Score"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

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

# -------------------- RESPONSE FORMAT --------------------
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

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
