import pandas as pd

# Load the parquet file
df = pd.read_parquet("../data/processed/fires_master.parquet")

print("data shape")
print(f"Total Fires: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

print("\ncolumn names")
print(df.columns.tolist())

print("\ndata types")
print(df.dtypes)

print("\ncountry check (Should only be ES and PT)")
print(df['country'].value_counts())

print("\nproduct era check (Should only be 0, 1, and 2)")
print(df['product_era'].value_counts().sort_index())

print("\nfirst 5 rows")
print(df.head(5))