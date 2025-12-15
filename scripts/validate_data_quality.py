import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "materials_features.csv")

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(" Dataset loaded successfully")
    return df

def summary_statistics(df):
    print("\n Summary Statistics:")
    print(df.describe())

    print("\n Missing Values:")
    print(df.isna().sum())

def plot_boxplots(df):
    numerical_cols = df.select_dtypes(include="number").columns

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[numerical_cols])
    plt.xticks(rotation=45)
    plt.title("Boxplots for Numerical Feature Validation")
    plt.tight_layout()
    plt.show()

def plot_heatmap(df):
    numerical_cols = df.select_dtypes(include="number").columns
    corr = df[numerical_cols].corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("\n Validating data quality...")
    df = load_data()
    summary_statistics(df)
    plot_boxplots(df)
    plot_heatmap(df)
    print("\n Data quality validation completed!")
