import pandas as pd

MATERIALS_PATH = "../data/materials_scaled.csv"
PRODUCTS_PATH = "../data/products_scaled.csv"


def load_data():
    materials = pd.read_csv(MATERIALS_PATH)
    products = pd.read_csv(PRODUCTS_PATH)
    return materials, products


def handle_missing_values(df, name):
    """Fill missing numeric values with mean, and categorical with mode."""
    print(f"\nChecking missing values for {name}:\n", df.isna().sum())

    # Numerical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    print(f"After handling missing values for {name}:\n", df.isna().sum())
    return df


def normalize_numerical(df):
    """Normalize numerical columns to 1-100 scale using Min-Max Scaling."""
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    
    for col in num_cols:
        min_val = df[col].min()
        max_val = df[col].max()

        if min_val != max_val:
            df[col] = ((df[col] - min_val) / (max_val - min_val)) * 99 + 1
        else:
            df[col] = 50  # constant column → mid value

    return df

def encode_categorical(df, name):
    """One-hot encode all categorical (string/object) columns."""
    cat_cols = df.select_dtypes(include=["object"]).columns

    print(f"\nEncoding categorical columns for {name}: {list(cat_cols)}")

    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    return df

def save_cleaned_data(materials, products):
    materials.to_csv("../data/materials_cleaned.csv", index=False)
    products.to_csv("../data/products_cleaned.csv", index=False)

    print("\nCleaned & Encoded files saved as:\n - materials_cleaned.csv\n - products_cleaned.csv")


if __name__ == "__main__":
    # 1 — Load data
    materials, products = load_data()

    # 2 — Missing value handling
    materials = handle_missing_values(materials, "Materials Dataset")
    products = handle_missing_values(products, "Products Dataset")
    
    detect_outliers(materials, "Materials Dataset")
    detect_outliers(products, "Products Dataset")
    # 3 — Normalize numeric columns
    materials = normalize_numerical(materials)
    products = normalize_numerical(products)

    # 4 — Encode categorical columns
    materials = encode_categorical(materials, "Materials Dataset")
    products = encode_categorical(products, "Products Dataset")

    # 5 — Save output
    save_cleaned_data(materials, products)

    print("\n Data cleaning, normalization & encoding completed successfully!")
