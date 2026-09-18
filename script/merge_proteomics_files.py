import pandas as pd

TUMOR_FILE = "proteomics_tumor.cct"
NORMAL_FILE = "proteomics_normal.cct"
OUTPUT_FILE = "merged_proteomics.csv"


def load_cct_file(filepath, label):

    df = pd.read_csv(filepath, sep="\t")

    id_column = df.columns[0]
    df = df.rename(columns={id_column: "Gene"})

    sample_columns = [c for c in df.columns if c != "Gene"]
    df = df.rename(columns={c: f"{label}_{c}" for c in sample_columns})

    return df


def merge_tumor_normal(tumor_df, normal_df):
    return pd.merge(tumor_df, normal_df, on="Gene", how="inner")


if __name__ == "__main__":

    tumor_df = load_cct_file(TUMOR_FILE, "Tumor")
    normal_df = load_cct_file(NORMAL_FILE, "Normal")

    print(f"Tumor: {len(tumor_df)} proteins, {len(tumor_df.columns) - 1} samples")
    print(f"Normal: {len(normal_df)} proteins, {len(normal_df.columns) - 1} samples")

    merged = merge_tumor_normal(tumor_df, normal_df)

    dropped = len(tumor_df) - len(merged)
    if dropped > 0:
        print(f"Note: {dropped} proteins excluded (not in both files).")

    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE} — sample columns now prefixed Tumor_/Normal_")