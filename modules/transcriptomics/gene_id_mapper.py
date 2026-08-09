import re
import pandas as pd

try:
    import mygene
    MYGENE_AVAILABLE = True
except ImportError:
    MYGENE_AVAILABLE = False


ENSEMBL_PATTERN = re.compile(r"^ENSG\d+")
ENTREZ_PATTERN = re.compile(r"^\d+$")


def detect_id_type(gene_ids, sample_size=50):
    """
    Returns 'ensembl', 'entrez', or 'symbol' based on a sample of
    gene ID values. 'symbol' means no conversion is needed.
    """

    sample = [str(g) for g in list(gene_ids)[:sample_size] if pd.notna(g)]

    if not sample:
        return "unknown"

    ensembl_count = sum(1 for g in sample if ENSEMBL_PATTERN.match(g))
    entrez_count = sum(1 for g in sample if ENTREZ_PATTERN.match(g))

    if ensembl_count / len(sample) > 0.8:
        return "ensembl"
    if entrez_count / len(sample) > 0.8:
        return "entrez"

    return "symbol"


class GeneIDConverter:
    """
    Converts Ensembl or Entrez gene IDs to HGNC gene symbols via
    the public MyGene.info service. Requires internet (same as
    Enrichment Analysis). Multiple IDs mapping to the same symbol
    are summed together — standard practice, verified against
    real duplicate-ID data before shipping.
    """

    def __init__(self, dataframe, gene_column):
        self.dataframe = dataframe
        self.gene_column = gene_column

    def is_ready(self):
        return MYGENE_AVAILABLE

    def convert(self, id_type, species="human"):

        if not MYGENE_AVAILABLE:
            return None, None, (
                "The 'mygene' package is not installed. "
                "Run: pip install mygene"
            )

        if id_type not in ("ensembl", "entrez"):
            return None, None, f"Unsupported ID type: {id_type}"

        scope = "ensembl.gene" if id_type == "ensembl" else "entrezgene"

        ids = self.dataframe[self.gene_column].astype(str).tolist()

        if id_type == "ensembl":
            ids = [i.split(".")[0] for i in ids]

        try:
            mg = mygene.MyGeneInfo()
            results = mg.querymany(
                ids, scopes=scope, fields="symbol",
                species=species, as_dataframe=False, verbose=False
            )
        except Exception as error:
            return None, None, f"Gene ID lookup failed: {error}"

        id_to_symbol = {
            entry["query"]: entry["symbol"]
            for entry in results
            if "symbol" in entry and "notfound" not in entry
        }

        working = self.dataframe.copy()
        working[self.gene_column] = ids
        working["Symbol"] = working[self.gene_column].map(id_to_symbol)

        n_total = len(working)
        n_mapped = int(working["Symbol"].notna().sum())

        working = working.dropna(subset=["Symbol"])

        sample_columns = [
            c for c in working.columns
            if c not in (self.gene_column, "Symbol")
        ]

        converted = working.groupby("Symbol", as_index=False)[sample_columns].sum()
        converted = converted.rename(columns={"Symbol": "Gene"})

        stats = {
            "Total IDs": n_total,
            "Successfully Mapped": n_mapped,
            "Unmapped (dropped)": n_total - n_mapped,
            "Final Gene Count (after merging duplicates)": len(converted)
        }

        return converted, stats, None

    def convert_using_annotation_file(
        self, annotation_filepath, id_column="GeneID", symbol_column="Symbol"
    ):
        """
        Offline conversion using NCBI's own annotation file instead
        of a live API call — instant, no internet needed after the
        one-time file download.
        """

        try:
            annotation = pd.read_csv(annotation_filepath, sep="\t")
        except Exception as error:
            return None, None, f"Could not read annotation file: {error}"

        if id_column not in annotation.columns or symbol_column not in annotation.columns:
            return None, None, (
                f"Annotation file must contain '{id_column}' and "
                f"'{symbol_column}' columns. Found: {list(annotation.columns)}"
            )

        id_to_symbol = dict(zip(
            annotation[id_column].astype(str), annotation[symbol_column]
        ))

        working = self.dataframe.copy()
        working[self.gene_column] = working[self.gene_column].astype(str)
        working["Symbol"] = working[self.gene_column].map(id_to_symbol)

        n_total = len(working)
        n_mapped = int(working["Symbol"].notna().sum())

        working = working.dropna(subset=["Symbol"])

        sample_columns = [
            c for c in working.columns if c not in (self.gene_column, "Symbol")
        ]

        converted = working.groupby("Symbol", as_index=False)[sample_columns].sum()
        converted = converted.rename(columns={"Symbol": "Gene"})

        stats = {
            "Total IDs": n_total,
            "Successfully Mapped": n_mapped,
            "Unmapped (dropped)": n_total - n_mapped,
            "Final Gene Count (after merging duplicates)": len(converted)
        }

        return converted, stats, None