import pandas as pd

from modules.transcriptomics.validator import (
    detect_gene_column,
    detect_sample_columns
)


class GeneFilter:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

        self.gene_column = detect_gene_column(self.df)

        self.samples = detect_sample_columns(self.df)

    def remove_zero_genes(self):

        self.df = self.df[
            self.df[self.samples].sum(axis=1) > 0
        ]

        return self.df

    def remove_low_count_genes(

        self,

        minimum_total_count=10

    ):

        self.df = self.df[
            self.df[self.samples].sum(axis=1)
            >= minimum_total_count
        ]

        return self.df

    def remove_sparse_genes(

        self,

        minimum_samples=2

    ):

        self.df = self.df[
            (
                self.df[self.samples] > 0
            ).sum(axis=1)
            >= minimum_samples
        ]

        return self.df

    def filter_by_cpm(self, min_cpm=1.0, min_samples=2):
        """
        Filters by CPM (counts per million) rather than raw count —
        the standard, library-size-aware approach (similar to
        edgeR's filterByExpr). A raw count threshold unfairly favors
        deeply-sequenced samples; this doesn't.
        """

        totals = self.df[self.samples].sum()
        cpm = (self.df[self.samples] / totals) * 1_000_000

        passes = (cpm >= min_cpm).sum(axis=1) >= min_samples

        self.df = self.df[passes]

        return self.df