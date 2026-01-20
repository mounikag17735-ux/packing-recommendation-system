import pandas as pd
import mysql.connector
from mysql.connector import Error

# STEP 1: LOAD DATA FROM CSV FILES

print("="*60)
print("STEP 1: LOADING DATA FROM CSV FILES")
print("="*60)

materials_df = pd.read_csv('data/materials_data.csv')
print("\n✓ Materials Data Loaded:")
print(materials_df)
print(f"\nShape: {materials_df.shape} (rows, columns)")

# Read product categories
categories_df = pd.read_csv('data/products_categories.csv')
print("\n✓ Product Categories Loaded:")
print(categories_df)
print(f"\nShape: {categories_df.shape} (rows, columns)")

# STEP 2: EXPLORE DATA

print("\n" + "="*60)
print("STEP 2: EXPLORING DATA")
print("="*60)

print("\nMaterials Data Info:")
print(materials_df.info())

print("\nMaterials Statistics:")
print(materials_df.describe())