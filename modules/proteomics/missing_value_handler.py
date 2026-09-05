import numpy as np
import pandas as pd


class MissingValueAnalyzer:
    """
    Proteomics missing values are NOT like RNA-seq zero counts.
    Low-abundance proteins are more likely to go undetected (below
    the instrument's detection limit) -- this is Missing Not At
    Random (MNAR), the standard understanding in the field (same
    logic used by Perseus, the most widely used companion tool to
    MaxQuant). Missing values carry real biological information
    ("likely very low or absent"), so imputing with something like
    the sample mean would be actively wrong.
    """

    def __init__(self, dataframe, protein_column, samples):
        self.df = dataframe
        self.protein_column = protein_column
        self.samples = samples

    def missingness_summary(self):

        values = self.df[self.samples]
        missing_fraction = values.isna().mean(axis=1)

        return pd.DataFrame({
            self.protein_column: self.df[self.protein_column],
            "Missing_Fraction": missing_fraction,
            "Valid_Count": values.notna().sum(axis=1),
            "Total_Samples": len(self.samples)
        })

    def per_group_detection(self, group_map):
        """
        Fraction of samples with a valid value, per protein, PER
        GROUP -- what filtering/imputation decisions should
        actually be based on, not overall missingness.
        """

        groups = sorted(set(group_map.values()))
        result = pd.DataFrame({self.protein_column: self.df[self.protein_column]})

        for group in groups:
            group_samples = [s for s in self.samples if group_map.get(s) == group]
            group_values = self.df[group_samples]
            result[f"Valid_Fraction_{group}"] = group_values.notna().mean(axis=1)

        return result


class ProteomicsFilter:
    """
    Filters by detection rate WITHIN groups, not overall -- a
    protein detected in 100% of Group A but 0% of Group B is
    exactly the kind of biologically interesting, all-or-nothing
    signal a naive overall filter would wrongly discard.
    """

    def __init__(self, dataframe, protein_column, samples):
        self.df = dataframe
        self.protein_column = protein_column
        self.samples = samples

    def filter_by_group_detection(self, group_map, min_valid_fraction=0.7):

        groups = sorted(set(group_map.values()))
        keep_mask = pd.Series(False, index=self.df.index)

        for group in groups:
            group_samples = [s for s in self.samples if group_map.get(s) == group]
            valid_fraction = self.df[group_samples].notna().mean(axis=1)
            keep_mask |= (valid_fraction >= min_valid_fraction)

        return self.df[keep_mask].reset_index(drop=True)


class Imputer:
    """
    Downshifted-normal-distribution imputation -- the standard
    method for MNAR proteomics data (Perseus default). Simulates
    "below detection limit" by drawing from a distribution shifted
    below and narrower than the protein's own observed values,
    rather than using the mean.

    Operates on log-transformed intensities.
    """

    def __init__(self, downshift=1.8, width=0.3, random_state=42):
        self.downshift = downshift
        self.width = width
        self.random_state = random_state

    def impute(self, dataframe, samples):

        rng = np.random.default_rng(self.random_state)

        df = dataframe.copy()
        values = df[samples].values.astype(float)

        row_means = np.nanmean(values, axis=1)
        row_stds = np.nanstd(values, axis=1)
        row_stds = np.where(np.isnan(row_stds) | (row_stds == 0), 0.5, row_stds)

        for i in range(values.shape[0]):

            missing_idx = np.isnan(values[i])
            n_missing = missing_idx.sum()

            if n_missing == 0:
                continue

            impute_mean = row_means[i] - self.downshift * row_stds[i]
            impute_std = row_stds[i] * self.width

            values[i, missing_idx] = rng.normal(impute_mean, impute_std, n_missing)

        df[samples] = values

        return df