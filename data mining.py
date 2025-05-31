
# Titanic Data Preprocessing

import pandas as pd
import numpy as np

# Load dataset (Ensure the 'train.csv' file is in the same directory)
df = pd.read_csv("train.csv")

# ------------------------------
# 1. Data Cleaning
# ------------------------------

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill missing Age with median
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Fill missing Embarked with mode
df.drop(columns=['Cabin'], inplace=True)  # Drop 'Cabin' due to too many missing values

# ------------------------------
# 2. Noisy Data Handling
# ------------------------------

# Apply binning to Age column to reduce noise
df['AgeBin'] = pd.cut(df['Age'], bins=5, labels=False)

# ------------------------------
# 3. Data Integration
# ------------------------------

# Load the test dataset for integration
try:
    df_test = pd.read_csv("test.csv")
    df['Dataset'] = 'train'
    df_test['Dataset'] = 'test'

    # Align columns for merging
    common_columns = list(set(df.columns).intersection(set(df_test.columns)))
    df_combined = pd.concat([df[common_columns], df_test[common_columns]], ignore_index=True)

    # Correlation matrix
    print("Correlation Matrix (Numeric Columns Only):")
    print(df_combined.corr(numeric_only=True))

except FileNotFoundError:
    print("test.csv not found. Skipping data integration step.")

# Save cleaned dataset
df.to_csv("cleaned_titanic_data.csv", index=False)

print("Preprocessing complete. Cleaned data saved as 'cleaned_titanic_data.csv'.")
