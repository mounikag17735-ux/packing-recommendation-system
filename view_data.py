import mysql.connector
from tabulate import tabulate


password = "vasu3107"  
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password=password,
    database="ecopack_data"
)


cursor = conn.cursor()


print("\n=== MATERIALS TABLE ===")
cursor.execute("SELECT * FROM materials LIMIT 10")
materials = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]
print(tabulate(materials, headers=columns, tablefmt="grid"))


print(f"\nTotal materials: ", end="")
cursor.execute("SELECT COUNT(*) FROM materials")
print(cursor.fetchone()[0])


print("\n\n=== PRODUCT CATEGORIES TABLE ===")
cursor.execute("SELECT * FROM product_categories")
categories = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]
print(tabulate(categories, headers=columns, tablefmt="grid"))

print(f"\nTotal categories: ", end="")
cursor.execute("SELECT COUNT(*) FROM product_categories")
print(cursor.fetchone()[0])

cursor.close()
conn.close()