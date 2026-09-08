"""
Merges LinkedOmics/CPTAC Tumor + Normal proteomics .cct files into
a single combined matrix, ready to upload into BioAI's Proteomics
module.

USAGE:
    python merge_proteomics_files.py

Edit the three variables below (TUMOR_FILE, NORMAL_FILE,
OUTPUT_FILE) to match your actual downloaded filenames first.
"""

import pandas as pd

TUMOR_FILE = "proteomics_tumor.cct"
NORMAL_FILE = "proteomics_normal.cct"
OUTPUT_FILE = "merged_proteomics.csv"


def load_cct_file(filepath):
    """
    Reads a LinkedOmics .cct file (tab-separated despite the
    unusual extension) and standardizes the gene/protein ID
    column to be named "Gene", regardless of its original header.
    """

    df = pd.read_csv(filepath, sep="\t")

    id_column = df.columns[0]
    df = df.rename(columns={id_column: "Gene"})

    return df


def merge_tumor_normal(tumor_df, normal_df):
    """
    Combines Tumor and Normal matrices into one, keeping only
    genes/proteins present in BOTH files (an inner join) -- this
    avoids introducing large blocks of missing values purely from
    gene-list mismatches between the two files, which is a
    separate concern from real biological missingness.
    """

    merged = pd.merge(tumor_df, normal_df, on="Gene", how="inner")

    return merged


if __name__ == "__main__":

    tumor_df = load_cct_file(TUMOR_FILE)
    normal_df = load_cct_file(NORMAL_FILE)

    print(f"Tumor file: {len(tumor_df)} proteins, "
          f"{len(tumor_df.columns) - 1} samples")
    print(f"Normal file: {len(normal_df)} proteins, "
          f"{len(normal_df.columns) - 1} samples")

    merged = merge_tumor_normal(tumor_df, normal_df)

    dropped = len(tumor_df) - len(merged)
    if dropped > 0:
        print(f"Note: {dropped} proteins were only in one file "
              "and were excluded (inner join).")

    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"\nMerged file saved: {OUTPUT_FILE}")
    print(f"Final shape: {len(merged)} proteins, "
          f"{len(merged.columns) - 1} total samples")