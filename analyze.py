import pandas as pd

# Load only the columns we need
s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("\n========== BASIC INFO ==========")

print("Source 1:", s1.shape)
print("Source 2:", s2.shape)
print("Source 3:", s3.shape)
print("Ground Truth:", gt.shape)

print("\n========== MISSING VALUES ==========")

print("Source 1:")
print(s1.isna().sum())

print("\nSource 2:")
print(s2.isna().sum())

print("\nSource 3:")
print(s3.isna().sum())

print("\n========== COUNTRIES ==========")

print("Source 1:")
print(s1["country"].value_counts())

print("\nSource 2:")
print(s2["country"].value_counts())

print("\nSource 3:")
print(s3["country"].value_counts())

print("\n========== GROUND TRUTH ==========")

print("Empty matches:")
print(
    gt["matched_entity_ids"]
    .fillna("")
    .eq("")
    .sum()
)

print("\nNumber of matches per Source 1:")

match_count = (
    gt["matched_entity_ids"]
    .fillna("")
    .apply(lambda x: 0 if x == "" else len(x.split(",")))
)

print(match_count.describe())

print("\nMatch count distribution:")
print(match_count.value_counts().sort_index())