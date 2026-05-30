import pandas as pd
# Notice the clean, package-style import here
from llmclean import clean_column, standardize_column, fill_missing, detect_anomalies

# Construct dummy corrupted raw pipeline matrix
data = {
    "name":    ["alice johnson", "BOB SMITH",  "Charlie  Brown", "DIANA prince", "eve"],
    "city":    ["new york",      "newyork",    "LA",             "los angeles",  "NYC"],
    "email":   ["alice@gmail.com","bob@email", "charlie@g.com",  None,           "eve@yahoo.com"],
    "country": [None,             "USA",       "United States",  "us",           None]
}

df = pd.DataFrame(data)

print("=" * 60)
print("INITIAL RAW MESSY DATASET")
print("=" * 60)
print(df)
print()

# Step 1: Normalize textual metrics
df["name"] = clean_column(
    df, "name",
    instruction="Convert this person's name to proper title case. Return only the name."
)

# Step 2: Bind categorical attributes to predefined domain boundaries
df["city"] = standardize_column(
    df, "city",
    categories=["New York", "Los Angeles", "Chicago", "Houston"]
)

# Step 3: Run contextual matrix heuristics to resolve missing fields
df["country"] = fill_missing(df, "country", context_columns=["city"])

# Step 4: Extract structural dataset outliers 
bad_emails = detect_anomalies(df, "email")

print("\n" + "=" * 60)
print("OPTIMIZED AND TRANSFORMED DATASET")
print("=" * 60)
print(df)
print(f"\n[ALERT] Identified Outliers inside 'email': {bad_emails}")
