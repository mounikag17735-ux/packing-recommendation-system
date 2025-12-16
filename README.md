📦EcoPackAI - AI-Powered Sustainable Packaging Recommendation System
Milestone 1 – Data Collection & Preprocessing
This project is part of the Infosys Springboard Virtual Internship (AI Domain). The goal is to build an intelligent system that recommends eco-friendly packaging materials based on sustainability metrics.

✅ Milestone 1 Completed Tasks
1️⃣ Data Import & Storage
Imported a 500-row synthetic eco-material dataset

Stored it inside a MySQL database (materials table)

Connected MySQL to Jupyter Notebook using mysql.connector

Loaded dataset into Pandas DataFrame

2️⃣ Data Quality Checks
Performed essential checks:

Dataset shape

Missing values

Duplicate rows

Descriptive statistics (df.describe())

3️⃣ Outlier Detection (IQR Method)
Identified outliers in:

Weight

Cost

CO₂ Emissions

Used IQR to calculate:

Q1, Q3

Lower bound

Upper bound

Outlier count per column

4️⃣ Outlier Treatment (Clipping)
df_clean[numeric_cols] = df[numeric_cols].clip(lower_bound, upper_bound, axis=1)

✔ All outliers were successfully removed.

5️⃣ Cleaned Dataset Export
Exported cleaned dataset as:

materials_cleaned.csv

This file will be used in further milestones.

📁 Project Structure
Packaging-Recommendation-System/ │ README.md │ requirements.txt │ ├── notebooks/ │ milestone1_data_preprocessing.ipynb │ ├── data/ │ materials_cleaned.csv

🛠️ Tools & Technologies Used
Python

Pandas

NumPy

Jupyter Notebook

VS Code

MySQL

Git & GitHub

🚀 Upcoming Milestone: Feature Engineering
Milestone 2 will include:

Sustainability Score

CO₂ Impact Index

Cost Efficiency Index

Durability Normalization

Correlation Heatmaps & EDA Visualizations

📞 Contact
Maintainer: Ayush Kumar Pandey Email: ayushpandey1974@gmail.com

GitHub: github.com/ayushpandey3357
