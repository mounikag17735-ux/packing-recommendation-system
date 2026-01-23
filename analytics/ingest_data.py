import pandas as pd
import sqlite3

DB_PATH = "data/materials.db"

def load_materials():
    df = pd.read_csv("data/processed/materials_scaled.csv")

    # Standardize materials columns
    df.columns = [
        "material_id",
        "material_type",
        "strength",
        "weight_capacity",
        "biodegradability_score",
        "co2_emission_score",
        "recyclability_percent",
        "industry_category"
    ]

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("materials", conn, if_exists="replace", index=False)
    conn.close()


def load_products():
    df = pd.read_csv("data/processed/products_scaled.csv")

    # Standardize products columns
    df.columns = [
        "product_id",
        "product_name",
        "category",
        "weight",
        "fragility_level",
        "industry"
    ]

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("products", conn, if_exists="replace", index=False)
    conn.close()


if __name__ == "__main__":
    load_materials()
    load_products()
