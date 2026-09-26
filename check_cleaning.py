import pandas as pd

df = pd.read_parquet("processed/train_source1_clean.parquet")

print(df[[
    "business_name",
    "business_name_clean",
    "business_address",
    "business_address_clean"
]].head(20).to_string(index=False))