print("🚀 Script started...")

# filter_karnataka.py

import pandas as pd
from pathlib import Path

SOURCE_FILE = "Soil data.csv"   # <-- make sure this matches your filename
OUTPUT_FILE = "soil_defaults.csv"

# 1) Load
path = Path(SOURCE_FILE)
if not path.exists():
    raise FileNotFoundError(f"Couldn't find {SOURCE_FILE} in the current folder.")

df = pd.read_csv(path)
print("\n✅ Loaded CSV. Here are the columns I found:\n", list(df.columns), "\n")

# 2) Rename columns to standard names
df = df.rename(columns={
    "District": "Location",
    "Nitrogen Value": "Nitrogen",
    "Phosphorous value": "Phosphorus",   # fixed spelling
    "Potassium value": "Potassium"
})

# 3) Keep only the needed columns
df = df[["Location", "Nitrogen", "Phosphorus", "Potassium"]]

# 4) Convert values to numeric (just in case)
for col in ["Nitrogen", "Phosphorus", "Potassium"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 5) Group by district (average values per district)
agg = df.groupby("Location", dropna=True)[["Nitrogen", "Phosphorus", "Potassium"]].mean().reset_index()

# 6) Round values for neatness
agg = agg.round({"Nitrogen": 2, "Phosphorus": 2, "Potassium": 2})

# 7) Save
agg.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved defaults to {OUTPUT_FILE}")
print("\nPreview:\n", agg.head(10))
