import pandas as pd

# File paths
MATERIALS_PATH = "../data/materials_cleaned.csv"
PRODUCTS_PATH = "../data/products_cleaned.csv"


def load_data():
    """Load cleaned datasets."""
    materials = pd.read_csv(MATERIALS_PATH)
    products = pd.read_csv(PRODUCTS_PATH)
    return materials, products


# ------------------------------------------------------------
# Feature 1: CO₂ Impact Index
# Lower emissions → Better score (1–100)
# ------------------------------------------------------------
def compute_co2_impact(df):
    if "Co2_EMISSION_SCORE" not in df.columns:
        print(" CO₂ column missing! Skipping CO₂ Impact Index.")
        return df
    
    co2 = df["Co2_EMISSION_SCORE"]
    df["CO2_Impact_Index"] = (1 - (co2 - co2.min()) / (co2.max() - co2.min())) * 99 + 1
    return df


# ------------------------------------------------------------
# Feature 2: Cost Efficiency Index
# Since we don't have cost, approximate with:
#    strength / weight_capacity
# More strength + lower weight capacity → more cost-efficient
# ------------------------------------------------------------
def compute_cost_efficiency(df):
    if "STRENGTH" not in df.columns or "WEIGHT_CAPACITY" not in df.columns:
        print(" Missing strength/weight columns. Skipping Cost Efficiency Index.")
        return df

    # Avoid division issues
    df["Cost_Efficiency_Index"] = df["STRENGTH"] / (df["WEIGHT_CAPACITY"] + 1e-6)

    # Normalize 1–100
    col = df["Cost_Efficiency_Index"]
    df["Cost_Efficiency_Index"] = ((col - col.min()) / (col.max() - col.min())) * 99 + 1
    
    return df


# ------------------------------------------------------------
# Feature 3: Material Suitability Score
# Custom weighted score:
#   strength (40%) +
#   recyclability (30%) +
#   biodegradability (30%)
# ------------------------------------------------------------
def compute_material_suitability(df):
    required = ["STRENGTH", "RECYCLABILITY_PERCENTAGE", "BIODEGRADABILITY_SCORE"]
    if not all(col in df.columns for col in required):
        print(" Missing fields for Material Suitability Score.")
        return df

    df["Material_Suitability"] = (
        df["STRENGTH"] * 0.4 +
        df["RECYCLABILITY_PERCENTAGE"] * 0.3 +
        df["BIODEGRADABILITY_SCORE"] * 0.3
    )
    return df


# ------------------------------------------------------------
# Save engineered features
# ------------------------------------------------------------
def save_features(materials, products):
    materials.to_csv("../data/materials_features.csv", index=False)
    products.to_csv("../data/products_features.csv", index=False)

    print("\nFeature-engineered datasets saved as:")
    print(" - materials_features.csv")
    print(" - products_features.csv")


# ------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------
if __name__ == "__main__":

    print("\n🔧 Loading cleaned datasets...")
    materials, products = load_data()

    print("\n Computing CO₂ Impact Index...")
    materials = compute_co2_impact(materials)

    print("\n Computing Cost Efficiency Index...")
    materials = compute_cost_efficiency(materials)

    print("\n Computing Material Suitability Score...")
    materials = compute_material_suitability(materials)

    # (Products dataset features can be added later if required)

    print("\n Saving engineered datasets...")
    save_features(materials, products)

    print("\n Feature engineering completed successfully!")
