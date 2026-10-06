import pandas as pd

# Load knowledge base
df = pd.read_csv("college_knowledge_base_dummy.csv")

# Display basic information
print("Dataset loaded successfully!")
print("Number of records:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nCategories:")
print(df["category"].value_counts())

print("\nFirst 5 records:")
print(df.head())