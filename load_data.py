import mysql.connector
import csv
import getpass

password = "vasu3107" 

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password=password,
    database="ecopack_data"
)

cursor = conn.cursor()


print("Loading materials data...")
with open('data/materials_data.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        cursor.execute("""
            INSERT INTO materials (
                material_id, material_name, material_type, strength_mpa, 
                weight_capacity_kg, biodegradability_score, co2_emission_score, 
                recyclability_percent, cost_per_kg
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            row['material_id'], row['material_name'], row['material_type'],
            row['strength_mpa'], row['weight_capacity_kg'], 
            row['biodegradability_score'], row['co2_emission_score'],
            row['recyclability_percent'], row['cost_per_kg']
        ))

print(f"Inserted {cursor.rowcount} rows into materials table")


print("Loading product categories data...")
with open('data/products_categories.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        cursor.execute("""
            INSERT INTO product_categories (
                category_id, category_name, protection_level, 
                temperature_sensitive, moisture_sensitive
            ) VALUES (%s, %s, %s, %s, %s)
        """, (
            row['category_id'], row['category_name'], row['protection_level'],
            row['temperature_sensitive'], row['moisture_sensitive']
        ))

print(f"Inserted {cursor.rowcount} rows into product_categories table")


conn.commit()


cursor.execute("SELECT COUNT(*) FROM materials")
materials_count = cursor.fetchone()[0]
print(f"\nTotal materials in database: {materials_count}")


cursor.execute("SELECT COUNT(*) FROM product_categories")
categories_count = cursor.fetchone()[0]
print(f"Total product categories in database: {categories_count}")


cursor.close()
conn.close()

print("\nData loaded successfully!")