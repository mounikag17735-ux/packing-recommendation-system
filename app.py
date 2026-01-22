from flask import Flask, jsonify, request, render_template, send_file
import joblib
import pandas as pd
import sqlite3
import os

import matplotlib
matplotlib.use("Agg")

from analytics.bi_metrics import get_bi_metrics
from analytics.bi_charts import generate_charts
from bi_dashboard.export_reports import load_logs, export_excel_report
from bi_dashboard.generate_pdf_report import generate_pdf
from init_db import init_db


# -------------------- PATH HELPER (CRITICAL FOR HF) --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def p(*paths):
    return os.path.join(BASE_DIR, *paths)


# -------------------- INIT DB --------------------
init_db()

API_KEY = os.getenv("API_KEY", "packaging_ai_2026_secret")
DB_PATH = p("data", "packaging.db")

app = Flask(__name__)

df = None
preprocessor = None
cost_model = None
co2_model = None
X = None


DEFAULT_MATERIAL = {
    "MATERIAL_TYPE": "Standard Packaging",
    "Predicted_Cost": None,
    "Predicted_CO2": None,
    "Final_Score": None,
    "Rank": 1,
    "Explanation": "Fallback recommendation due to insufficient matching data"
}


# -------------------- DB --------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_materials_from_db():
    conn = get_db_connection()
    try:
        df_local = pd.read_sql("SELECT * FROM materials", conn)
        print(f"✅ Loaded {len(df_local)} materials from DB")
        return df_local
    finally:
        conn.close()


# -------------------- MODEL LOADER --------------------
def load_models():
    global preprocessor, cost_model, co2_model, X

    if preprocessor is not None:
        return True

    try:
        preprocessor = joblib.load(p("artifacts", "preprocessor.pkl"))
        X = joblib.load(p("artifacts", "X.pkl"))

        cost_model = joblib.load(p("artifacts", "cost_model.pkl"))
        co2_model = joblib.load(p("artifacts", "co2_model.pkl"))

        print("✅ ML models loaded")
        return True

    except Exception as e:
        print("⚠️ ML models not available:", e)
        return False


# -------------------- ROUTES --------------------
@app.route("/")
def ui():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    generate_charts()
    metrics = get_bi_metrics()
    return render_template("dashboard.html", metrics=metrics)


@app.route("/export/excel")
def export_excel():
    df_logs = load_logs()
    if df_logs.empty:
        return jsonify({"error": "No data available"}), 400

    export_excel_report(df_logs)
    latest_file = sorted(os.listdir(p("reports")))[-1]
    return send_file(p("reports", latest_file), as_attachment=True)


@app.route("/export/pdf")
def export_pdf():
    path = generate_pdf()
    return send_file(path, as_attachment=True)


@app.route("/recommend", methods=["POST"])
def recommend_material():
    if request.headers.get("X-API-KEY") != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    product_input = request.get_json()
    recommendations_df = generate_ai_recommendations(product_input)
    formatted = format_recommendation_response(recommendations_df)

    return jsonify({
        "status": "success",
        "recommendation_count": len(formatted),
        "recommended_materials": formatted
    })


# -------------------- AI LOGIC --------------------
def generate_ai_recommendations(product_input, top_n=5):
    global df

    if df is None:
        df = load_materials_from_db()

    if df.empty or not load_models():
        return pd.DataFrame([DEFAULT_MATERIAL])

    eco_priority = float(product_input.get("eco_priority", 0.5))
    fragility = product_input.get("fragility_level", "medium").lower()
    product_weight = float(product_input.get("product_weight", 0))
    industry = product_input.get("industry", "").lower()

    strength_map = {"low": 40, "medium": 60, "high": 80}
    min_strength = strength_map.get(fragility, 60)

    filtered_df = df[
        (df["STRENGTH"] >= min_strength) &
        (df["WEIGHT_CAPACITY"] >= product_weight) &
        (df["INDUSTRY_CATEGORY"].str.lower().str.contains(industry, na=False))
    ].copy()

    if filtered_df.empty:
        return pd.DataFrame([DEFAULT_MATERIAL])

    X_filtered = X.loc[filtered_df.index]
    X_processed = preprocessor.transform(X_filtered)

    filtered_df["Predicted_Cost"] = cost_model.predict(X_processed)
    filtered_df["Predicted_CO2"] = co2_model.predict(X_processed)

    cost_norm = filtered_df["Predicted_Cost"].max() - filtered_df["Predicted_Cost"].min() or 1
    co2_norm = filtered_df["Predicted_CO2"].max() - filtered_df["Predicted_CO2"].min() or 1

    filtered_df["Final_Score"] = (
        (1 - eco_priority) *
        (filtered_df["Predicted_Cost"] - filtered_df["Predicted_Cost"].min()) / cost_norm +
        eco_priority *
        (filtered_df["Predicted_CO2"] - filtered_df["Predicted_CO2"].min()) / co2_norm
    )

    filtered_df["Rank"] = filtered_df["Final_Score"].rank(method="first").astype(int)
    filtered_df["Explanation"] = "Balanced cost & sustainability"

    return filtered_df.sort_values("Rank").head(top_n).reset_index(drop=True)


# -------------------- RESPONSE --------------------
def safe_round(val, digits=2):
    try:
        if val is None or pd.isna(val):
            return None
        return round(float(val), digits)
    except:
        return None


def format_recommendation_response(df):
    results = []

    for _, row in df.iterrows():
        results.append({
            "material_name": row.get("MATERIAL_TYPE", "Standard Packaging"),
            "rank": int(row.get("Rank", 1)),
            "predicted_cost": safe_round(row.get("Predicted_Cost")),
            "predicted_co2": safe_round(row.get("Predicted_CO2")),
            "final_score": safe_round(row.get("Final_Score"), 4),
            "explanation": row.get(
                "Explanation",
                "Fallback recommendation due to insufficient matching data"
            )
        })

    return results


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
