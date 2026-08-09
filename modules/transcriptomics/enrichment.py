import pandas as pd

try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False


ENRICHR_LIBRARIES = {
    "GO Biological Process": "GO_Biological_Process_2023",
    "GO Molecular Function": "GO_Molecular_Function_2023",
    "GO Cellular Component": "GO_Cellular_Component_2023",
    "KEGG Pathways": "KEGG_2021_Human",
    "Reactome Pathways": "Reactome_2022"
}

SUPPORTED_ORGANISMS = ["Human", "Mouse", "Yeast", "Fly", "Fish", "Worm"]


class EnrichmentAnalysis:
    """
    Over-representation analysis against GO / KEGG / Reactome via
    the public Enrichr web service (gseapy). Requires an active
    internet connection on the machine running the app — no
    account or API key needed.
    """

    def __init__(self, gene_list):

        cleaned = []
        seen = set()

        for gene in gene_list:
            if gene is None:
                continue
            gene_str = str(gene).strip()
            if not gene_str or gene_str.lower() == "nan":
                continue
            if gene_str not in seen:
                seen.add(gene_str)
                cleaned.append(gene_str)

        self.gene_list = cleaned

    def is_ready(self):
        return GSEAPY_AVAILABLE and len(self.gene_list) >= 3

    def run(self, library_label, organism="Human", fdr_threshold=0.05):
        """
        Returns (results_dataframe, error_message).
        Exactly one of the two will be None — the UI only needs
        to check `if error:` to know whether to display results.
        """

        if not GSEAPY_AVAILABLE:
            return None, (
                "The 'gseapy' package is not installed. "
                "Run: pip install gseapy"
            )

        if len(self.gene_list) < 3:
            return None, (
                "At least 3 genes are required to run enrichment "
                "analysis. Adjust your Differential Expression "
                "thresholds to include more significant genes."
            )
        max_genes = 2000

        if len(self.gene_list) > max_genes:
            self.gene_list = self.gene_list[:max_genes]

        library_name = ENRICHR_LIBRARIES.get(library_label)

        if library_name is None:
            return None, f"Unknown gene set library: {library_label}"

        try:
            enrichment = gp.enrichr(
                gene_list=self.gene_list,
                gene_sets=[library_name],
                organism=organism.lower(),
                outdir=None,
                no_plot=True
            )
        except Exception as error:
            return None, (
                "Enrichment request failed. This step requires an "
                "active internet connection to reach Enrichr; if "
                "you're online, the detail below may point to the "
                f"cause. Details: {error}"
            )

        if (
            enrichment is None
            or enrichment.results is None
            or enrichment.results.empty
        ):
            return None, (
                "No enrichment results were returned. This can "
                "happen if none of your genes matched known gene "
                "sets, or the gene symbols are not standard HGNC "
                "symbols."
            )

        results = enrichment.results.copy()

        results = results.rename(columns={
            "Term": "Pathway",
            "Overlap": "Gene_Overlap",
            "P-value": "P_Value",
            "Adjusted P-value": "Adj_P_Value",
            "Genes": "Matched_Genes",
            "Combined Score": "Combined_Score"
        })

        keep_columns = [
            "Pathway", "Gene_Overlap", "P_Value",
            "Adj_P_Value", "Combined_Score", "Matched_Genes"
        ]

        results = results[[c for c in keep_columns if c in results.columns]]

        results["Significant"] = results["Adj_P_Value"] < fdr_threshold

        results = results.sort_values("Adj_P_Value").reset_index(drop=True)

        return results, None