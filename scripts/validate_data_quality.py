import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "materials_cleaned.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(" BASIC DATA OVERVIEW")
print(df.info())

print("\n SUMMARY STATISTICS")
print(df.describe())

print("\n MISSING VALUES")
print(df.isna().sum())
