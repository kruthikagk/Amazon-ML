import pandas as pd

files = {
    "train_source1.tsv": "dataset/train/train_source1.tsv",
    "train_source2.tsv": "dataset/train/train_source2.tsv",
    "train_source3.tsv": "dataset/train/train_source3.tsv"
}

for name, path in files.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    df = pd.read_csv(path, sep="\t")

    print("\nNumber of rows:")
    print(len(df))

    print("\nCountry distribution:")
    print(df["country"].value_counts())

    print("\nMissing business_name:")
    print(df["business_name"].isna().sum())

    print("\nMissing business_address:")
    print(df["business_address"].isna().sum())

    print("\nUnique business_name:")
    print(df["business_name"].nunique())

    print("\nUnique business_address:")
    print(df["business_address"].nunique())