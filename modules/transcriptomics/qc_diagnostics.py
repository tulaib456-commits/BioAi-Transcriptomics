import numpy as np
import pandas as pd

from modules.transcriptomics.validator import (
    detect_gene_column,
    detect_sample_columns
)


class QCDiagnostics:
    """
    Standard first-pass RNA-seq QC diagnostics: library sizes,
    sample-sample correlation (for outlier/batch-effect detection),
    and top expressed genes. Every one of these is a step a
    reviewer expects to see before any downstream analysis.
    """

    def __init__(self, dataframe):
        self.df = dataframe
        self.gene_column = detect_gene_column(dataframe)
        self.samples = detect_sample_columns(dataframe)

    def library_sizes(self):
        return self.df[self.samples].sum().to_dict()

    def log_cpm(self):
        totals = self.df[self.samples].sum()
        cpm = (self.df[self.samples] / totals) * 1_000_000
        return np.log2(cpm + 1)

    def correlation_matrix(self):

        log_cpm = self.log_cpm()

        corr_values = np.corrcoef(log_cpm.values.T)

        return pd.DataFrame(
            corr_values, index=log_cpm.columns, columns=log_cpm.columns
        )

    def flag_outlier_samples(self, threshold=2.0):
        """
        Flags samples whose mean correlation to all other samples
        is more than `threshold` standard deviations below the
        average — a simple, standard outlier heuristic.
        """

        corr = self.correlation_matrix()
        n = len(self.samples)

        mean_corr = (corr.sum(axis=1) - 1) / (n - 1)

        z_scores = (mean_corr - mean_corr.mean()) / mean_corr.std()

        flagged = mean_corr[z_scores < -threshold]

        return mean_corr.sort_values(), list(flagged.index)

    def top_expressed_genes(self, n=10):

        totals = self.df[self.samples].sum(axis=1)

        top = self.df.assign(Total_Expression=totals).sort_values(
            "Total_Expression", ascending=False
        ).head(n)

        return top[[self.gene_column, "Total_Expression"]]