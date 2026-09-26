import pandas as pd
import re
import unicodedata
import os


# ============================================================
# 1. GENERAL TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    General normalization:
    - Handles missing values
    - Unicode normalization
    - Lowercase
    - Punctuation normalization
    - Extra whitespace removal
    """

    if pd.isna(value):
        return ""

    value = str(value)

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Lowercase
    value = value.lower()

    # Replace common separators with spaces
    value = value.replace("&", " and ")

    # Remove punctuation/special characters
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# 2. BUSINESS NAME NORMALIZATION
# ============================================================

def normalize_name(value):

    value = normalize_text(value)

    if not value:
        return ""

    # Carefully selected business/legal abbreviations
    name_replacements = {
        r"\bpvt\b": "private",
        r"\bpvtltd\b": "private limited",
        r"\bpvt ltd\b": "private limited",
        r"\bpriv ltd\b": "private limited",

        r"\bltd\b": "limited",
        r"\blimited\b": "limited",

        r"\binc\b": "incorporated",
        r"\bcorp\b": "corporation",
        r"\bco\b": "company",

        r"\bllc\b": "limited liability company",
        r"\bllp\b": "limited liability partnership",

        r"\bplc\b": "public limited company"
    }

    for pattern, replacement in name_replacements.items():
        value = re.sub(pattern, replacement, value)

    # Remove repeated whitespace created by replacements
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# 3. ADDRESS NORMALIZATION
# ============================================================

def normalize_address(value):

    value = normalize_text(value)

    if not value:
        return ""

    # Common address abbreviations
    address_replacements = {
        r"\brd\b": "road",
        r"\brd\.\b": "road",

        r"\bst\b": "street",
        r"\bst\.\b": "street",

        r"\bave\b": "avenue",
        r"\bave\.\b": "avenue",

        r"\bblvd\b": "boulevard",
        r"\bblvd\.\b": "boulevard",

        r"\bdr\b": "drive",
        r"\bdr\.\b": "drive",

        r"\bln\b": "lane",
        r"\bln\.\b": "lane",

        r"\bhwy\b": "highway",
        r"\bhwy\.\b": "highway",

        r"\bpkwy\b": "parkway",
        r"\bpkwy\.\b": "parkway",

        r"\bct\b": "court",
        r"\bct\.\b": "court",

        r"\bcir\b": "circle",
        r"\bcir\.\b": "circle",

        r"\bapt\b": "apartment",
        r"\bapt\.\b": "apartment",

        r"\bste\b": "suite",
        r"\bste\.\b": "suite",

        r"\bfl\b": "floor",
        r"\bfl\.\b": "floor",

        r"\bhno\b": "house number",
        r"\bh no\b": "house number",

        r"\bno\b": "number"
    }

    for pattern, replacement in address_replacements.items():
        value = re.sub(pattern, replacement, value)

    # Normalize whitespace again
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# 4. PROCESS A FILE
# ============================================================

def clean_file(input_file, output_file):

    print(f"\nReading: {input_file}")

    df = pd.read_csv(input_file, sep="\t")

    # General normalized text
    df["business_name_clean"] = df["business_name"].apply(
        normalize_name
    )

    df["business_address_clean"] = df["business_address"].apply(
        normalize_address
    )

    # Save as Parquet
    df.to_parquet(
        output_file,
        index=False
    )

    print(f"Saved: {output_file}")
    print(f"Rows: {len(df):,}")


# ============================================================
# 5. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("processed", exist_ok=True)


# ============================================================
# 6. CLEAN TRAINING DATA
# ============================================================

clean_file(
    "dataset/train/train_source1.tsv",
    "processed/train_source1_clean.parquet"
)

clean_file(
    "dataset/train/train_source2.tsv",
    "processed/train_source2_clean.parquet"
)

clean_file(
    "dataset/train/train_source3.tsv",
    "processed/train_source3_clean.parquet"
)


print("\n===================================")
print("CLEANING COMPLETED SUCCESSFULLY")
print("===================================")