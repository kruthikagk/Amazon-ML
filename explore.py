import pandas as pd

source1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
source2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
source3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
ground_truth = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

print("\nSOURCE 1")
print(source1.head())
print("Shape:", source1.shape)
print("Columns:", source1.columns.tolist())

print("\nSOURCE 2")
print(source2.head())
print("Shape:", source2.shape)
print("Columns:", source2.columns.tolist())

print("\nSOURCE 3")
print(source3.head())
print("Shape:", source3.shape)
print("Columns:", source3.columns.tolist())

print("\nGROUND TRUTH")
print(ground_truth.head())
print("Shape:", ground_truth.shape)
print("Columns:", ground_truth.columns.tolist())