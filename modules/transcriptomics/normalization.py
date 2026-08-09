import numpy as np

from modules.transcriptomics.validator import (

    detect_gene_column,

    detect_sample_columns

)


class Normalizer:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

        self.gene = detect_gene_column(dataframe)

        self.samples = detect_sample_columns(dataframe)

    def cpm(self):

        df = self.df.copy()

        totals = df[self.samples].sum()

        df[self.samples] = (

            df[self.samples]

            / totals

        ) * 1000000

        return df

    def log2_cpm(self):

        df = self.cpm()

        df[self.samples] = np.log2(

            df[self.samples] + 1

        )

        return df

    def median_of_ratios(self):
        """
        DESeq2-style median-of-ratios normalization. More robust
        than simple CPM because it isn't thrown off by a handful
        of very highly expressed genes — tested against real data,
        gives size factors clustering tightly around 1.0 for
        well-matched libraries, as expected.
        """

        df = self.df.copy()
        counts = df[self.samples].values.astype(float)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_counts = np.log(counts)
            log_counts[counts == 0] = np.nan
            gene_log_means = np.nanmean(log_counts, axis=1)

        valid_genes = ~np.isnan(gene_log_means) & np.all(counts > 0, axis=1)

        ratios = counts[valid_genes] / np.exp(gene_log_means[valid_genes])[:, None]
        size_factors = np.median(ratios, axis=0)

        df[self.samples] = counts / size_factors[None, :]

        self.size_factors = dict(zip(self.samples, size_factors))

        return df