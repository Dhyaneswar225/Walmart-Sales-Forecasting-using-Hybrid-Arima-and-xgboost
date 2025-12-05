import pandas as pd

# -------------------------
# Step 3: Load CSV Files
# -------------------------

train_path     = "F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\data\\train.csv"
features_path  = "F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\data\\features.csv"
stores_path    = "F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\data\\stores.csv"

train_df    = pd.read_csv(train_path, parse_dates=["Date"])
features_df = pd.read_csv(features_path, parse_dates=["Date"])
stores_df   = pd.read_csv(stores_path)

# -------------------------
# Step 4: Merge datasets
# -------------------------

# Merge train + features on Store, Date
df = train_df.merge(features_df, on=["Store", "Date"], how="left")
df["IsHoliday"] = df["IsHoliday_x"]
df = df.drop(columns=["IsHoliday_x", "IsHoliday_y"])

# Merge store metadata
df = df.merge(stores_df, on="Store", how="left")


# Sort chronologically
df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

# -------------------------
# Step 5: Clean missing values
# -------------------------

# MarkDown1–5 → fill NaN with 0
markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
for col in markdown_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# CPI + Unemployment → forward fill
for col in ["CPI", "Unemployment"]:
    if col in df.columns:
        df[col] = df[col].ffill()

# Temperature → store-wise mean fill
if "Temperature" in df.columns:
    df["Temperature"] = df.groupby("Store")["Temperature"].transform(
        lambda x: x.fillna(x.mean())
    )

# Final dataset
print(df.head())
print(df.isna().sum())

output_path = "F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\data\\merged_walmart.csv"
df.to_csv(output_path, index=False)

print("\nMerged dataset saved to:")
print(output_path)
print("Shape:", df.shape)
