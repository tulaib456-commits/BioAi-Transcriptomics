import numpy as np
import pandas as pd


class ProteomicsNormalizer:
    """
    Proteomics intensities need different normalization than
    RNA-seq counts -- there's no "library size" concept here.
    """

    def __init__(self, dataframe, samples):
        self.df = dataframe
        self.samples = samples

    def median_normalization(self):
        """
        Shifts each sample so its median matches the grand median
        -- corrects systematic per-run intensity shifts (a real,
        common LC-MS artifact) without distorting each sample's
        distribution shape.
        """

        df = self.df.copy()
        values = df[self.samples].astype(float)

        sample_medians = values.median()
        grand_median = sample_medians.median()
        shift = grand_median - sample_medians

        df[self.samples] = values + shift

        return df

    def quantile_normalization(self):
        """
        Forces every sample to an identical intensity distribution
        -- a stronger correction, standard when runs show more
        complex technical variation.
        """

        df = self.df.copy()
        values = df[self.samples].astype(float)

        sorted_values = np.sort(values.values, axis=0)
        row_means = sorted_values.mean(axis=1)
        ranks = values.rank(method="average").values

        normalized = np.zeros_like(values.values)

        for col in range(values.shape[1]):
            normalized[:, col] = np.interp(
                ranks[:, col],
                np.arange(1, len(row_means) + 1),
                row_means
            )

        df[self.samples] = normalized

        return df