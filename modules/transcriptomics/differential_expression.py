import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.multitest import multipletests

from modules.transcriptomics.validator import (
    detect_gene_column,
    detect_sample_columns
)


def suggest_groups(samples):
    """
    Try to auto-split sample names into two groups by finding a
    common underscore-separated token that takes exactly 2 distinct
    values across all samples (e.g. 'mock' vs 'CoV2').

    Returns (group_a, group_b) lists of sample names, or (None, None)
    if no clean 2-way split is found.
    """

    tokenized = [name.split("_") for name in samples]

    max_tokens = max(len(tokens) for tokens in tokenized)

    for position in range(max_tokens):

        values = []

        for tokens in tokenized:

            if position < len(tokens):
                values.append(tokens[position])
            else:
                values.append(None)

        unique_values = sorted(set(values))

        if len(unique_values) == 2 and None not in unique_values:

            group_a = [
                sample
                for sample, value in zip(samples, values)
                if value == unique_values[0]
            ]

            group_b = [
                sample
                for sample, value in zip(samples, values)
                if value == unique_values[1]
            ]

            return group_a, group_b

    return None, None


class DifferentialExpression:
    """
    Runs gene-by-gene differential expression between two groups
    of samples on RAW (filtered, non-normalized) counts.

    Internally applies CPM + log2 normalization so results do not
    depend on whatever normalization method was chosen elsewhere
    in the app for export purposes.
    """

    def __init__(self, dataframe, group_a, group_b):

        self.gene_column = detect_gene_column(dataframe)

        self.group_a = group_a
        self.group_b = group_b

        self.raw = dataframe.copy()

        self.df = self._cpm_log2(dataframe)

    def _cpm_log2(self, dataframe):

        df = dataframe.copy()

        all_samples = self.group_a + self.group_b

        totals = df[all_samples].sum()

        df[all_samples] = (df[all_samples] / totals) * 1_000_000

        df[all_samples] = np.log2(df[all_samples] + 1)

        return df

    def run(self, fdr_threshold=0.05, log2fc_threshold=1.0):

        genes = self.df[self.gene_column].values

        a_values = self.df[self.group_a].values.astype(float)
        b_values = self.df[self.group_b].values.astype(float)

        mean_a = a_values.mean(axis=1)
        mean_b = b_values.mean(axis=1)

        log2fc = mean_b - mean_a

        t_stats, p_values = stats.ttest_ind(
            b_values,
            a_values,
            axis=1,
            equal_var=False
        )

        p_values = np.nan_to_num(p_values, nan=1.0)

        _, adjusted_p_values, _, _ = multipletests(
            p_values,
            method="fdr_bh"
        )

        results = pd.DataFrame({
            "Gene": genes,
            "Mean_GroupA": mean_a,
            "Mean_GroupB": mean_b,
            "Log2FC": log2fc,
            "P_Value": p_values,
            "Adj_P_Value": adjusted_p_values
        })

        results["Significant"] = (
            (results["Adj_P_Value"] < fdr_threshold)
            & (results["Log2FC"].abs() >= log2fc_threshold)
        )

        results = results.sort_values(
            "Adj_P_Value"
        ).reset_index(drop=True)

        return results
