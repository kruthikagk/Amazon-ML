import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH = PROJECT_ROOT / "data" / "train_ground_truth.tsv"
OUTPUT = PROJECT_ROOT / "output" / "ground_truth_analysis.txt"

print("Loading ground truth...")
print(f"File: {GROUND_TRUTH}")

df = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("\n=== BASIC INFORMATION ===")
print(f"Row count: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")

print("\n=== SCHEMA ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isna().sum())

print("\n=== EMPTY VALUES ===")
for column in df.columns:
    empty = df[column].astype(str).str.strip().eq("").sum()
    print(f"{column}: {empty:,}")

print("\n=== DUPLICATE SOURCE1 IDs ===")
duplicate_ids = df["source1_entity_id"].duplicated().sum()
unique_ids = df["source1_entity_id"].nunique()

print(f"Duplicate rows: {duplicate_ids:,}")
print(f"Unique Source1 IDs: {unique_ids:,}")


def count_matches(value):
    value = str(value).strip()

    if value == "":
        return 0

    return len(value.split(","))


df["match_count"] = df["matched_entity_ids"].apply(count_matches)

zero_matches = (df["match_count"] == 0).sum()
one_match = (df["match_count"] == 1).sum()
two_matches = (df["match_count"] == 2).sum()
three_plus = (df["match_count"] >= 3).sum()

total_relationships = df["match_count"].sum()
mean_matches = df["match_count"].mean()
median_matches = df["match_count"].median()
max_matches = df["match_count"].max()

print("\n=== MATCH DISTRIBUTION ===")
print(f"0 matches:  {zero_matches:,}")
print(f"1 match:    {one_match:,}")
print(f"2 matches:  {two_matches:,}")
print(f"3+ matches: {three_plus:,}")

print("\n=== MATCH STATISTICS ===")
print(f"Total matched relationships: {total_relationships:,}")
print(f"Mean matches per Source1: {mean_matches:.4f}")
print(f"Median matches per Source1: {median_matches:.0f}")
print(f"Maximum matches per Source1: {max_matches}")

total = len(df)

print("\n=== PERCENTAGES ===")
print(f"0 matches:  {zero_matches / total * 100:.2f}%")
print(f"1 match:    {one_match / total * 100:.2f}%")
print(f"2 matches:  {two_matches / total * 100:.2f}%")
print(f"3+ matches: {three_plus / total * 100:.2f}%")

print("\n=== DETAILED DISTRIBUTION ===")
distribution = df["match_count"].value_counts().sort_index()

for count, frequency in distribution.items():
    print(f"{count} matches: {frequency:,}")


# Save results separately
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("GROUND TRUTH ANALYSIS\n")
    f.write("=" * 60 + "\n\n")

    f.write("DATASET OVERVIEW\n")
    f.write("-" * 60 + "\n")
    f.write(f"Row count: {len(df):,}\n")
    f.write(f"Columns: {df.columns.tolist()}\n")
    f.write(f"Unique Source1 IDs: {unique_ids:,}\n\n")

    f.write("SCHEMA\n")
    f.write("-" * 60 + "\n")
    for column, dtype in df.dtypes.items():
        f.write(f"{column}: {dtype}\n")

    f.write("\nMISSING VALUES\n")
    f.write("-" * 60 + "\n")
    for column in df.columns:
        f.write(f"{column}: {df[column].isna().sum():,}\n")

    f.write("\nEMPTY VALUES\n")
    f.write("-" * 60 + "\n")
    for column in df.columns:
        empty = df[column].astype(str).str.strip().eq("").sum()
        f.write(f"{column}: {empty:,}\n")

    f.write("\nDUPLICATE SOURCE1 IDs\n")
    f.write("-" * 60 + "\n")
    f.write(f"Duplicate rows: {duplicate_ids:,}\n")

    f.write("\nMATCH DISTRIBUTION\n")
    f.write("-" * 60 + "\n")
    f.write(f"0 matches: {zero_matches:,}\n")
    f.write(f"1 match: {one_match:,}\n")
    f.write(f"2 matches: {two_matches:,}\n")
    f.write(f"3+ matches: {three_plus:,}\n")

    f.write("\nMATCH STATISTICS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Total matched relationships: {total_relationships:,}\n")
    f.write(f"Mean matches per Source1: {mean_matches:.4f}\n")
    f.write(f"Median matches per Source1: {median_matches:.0f}\n")
    f.write(f"Maximum matches per Source1: {max_matches}\n")

    f.write("\nPERCENTAGES\n")
    f.write("-" * 60 + "\n")
    f.write(f"0 matches: {zero_matches / total * 100:.2f}%\n")
    f.write(f"1 match: {one_match / total * 100:.2f}%\n")
    f.write(f"2 matches: {two_matches / total * 100:.2f}%\n")
    f.write(f"3+ matches: {three_plus / total * 100:.2f}%\n")

    f.write("\nDETAILED DISTRIBUTION\n")
    f.write("-" * 60 + "\n")

    for count, frequency in distribution.items():
        f.write(f"{count} matches: {frequency:,}\n")

print(f"\nAnalysis saved to:")
print(OUTPUT)