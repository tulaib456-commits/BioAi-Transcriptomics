import numpy as np
import pandas as pd
from scipy import stats


def suggest_multi_groups(samples, max_groups=8):
    """
    Auto-detects a grouping variable with 2-8 distinct values from
    underscore-separated sample name tokens (e.g. tissue type,
    treatment condition). Returns {sample: group_label} or None.
    """

    tokenized = [name.split("_") for name in samples]
    max_tokens = max(len(tokens) for tokens in tokenized)

    for position in range(max_tokens):

        values = []

        for tokens in tokenized:
            values.append(tokens[position] if position < len(tokens) else None)

        unique_values = sorted(set(values))

        if None not in unique_values and 2 <= len(unique_values) <= max_groups:
            return {sample: value for sample, value in zip(samples, values)}

    return None


class MultiGroupDE:
    """
    One-way ANOVA across 2+ groups, per gene, fully vectorized
    across all genes at once (no per-gene Python loop — tested at
    ~28,000 genes in 0.02s vs 23s for the naive loop version).
    Runs on CPM + log2 normalized values, computed internally so
    results don't depend on whichever normalization method was
    picked in the Normalization tab.
    """

    def __init__(self, dataframe, group_map, gene_column):
        self.gene_column = gene_column
        self.group_map = group_map
        self.samples = list(group_map.keys())
        self.groups = sorted(set(group_map.values()))
        self.df = self._cpm_log2(dataframe)

    def _cpm_log2(self, dataframe):
        df = dataframe.copy()
        totals = df[self.samples].sum()
        df[self.samples] = (df[self.samples] / totals) * 1_000_000
        df[self.samples] = np.log2(df[self.samples] + 1)
        return df

    def run(self, fdr_threshold=0.05, min_effect_size=0.0):

        from statsmodels.stats.multitest import multipletests

        genes = self.df[self.gene_column].values
        values = self.df[self.samples].values.astype(float)

        sample_index = {s: i for i, s in enumerate(self.samples)}

        group_sample_lists = {
            group: [s for s in self.samples if self.group_map[s] == group]
            for group in self.groups
        }

        k = len(self.groups)
        n_total = values.shape[1]

        grand_mean = values.mean(axis=1)

        ss_between = np.zeros(values.shape[0])
        ss_within = np.zeros(values.shape[0])
        group_means = {}

        for group in self.groups:

            idx = [sample_index[s] for s in group_sample_lists[group]]
            group_vals = values[:, idx]
            n_g = group_vals.shape[1]
            mean_g = group_vals.mean(axis=1)

            group_means[group] = mean_g
            ss_between += n_g * (mean_g - grand_mean) ** 2
            ss_within += ((group_vals - mean_g[:, None]) ** 2).sum(axis=1)

        df_between = k - 1
        df_within = n_total - k

        ms_between = ss_between / df_between
        ms_within = np.where(df_within > 0, ss_within / df_within, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            f_stats = np.where(ms_within > 0, ms_between / ms_within, 0.0)

        p_values = stats.f.sf(f_stats, df_between, df_within)
        p_values = np.nan_to_num(p_values, nan=1.0)
        f_stats = np.nan_to_num(f_stats, nan=0.0)

        _, adjusted_p_values, _, _ = multipletests(p_values, method="fdr_bh")

        means_matrix = np.column_stack([group_means[g] for g in self.groups])
        max_mean_diff = means_matrix.max(axis=1) - means_matrix.min(axis=1)

        results = pd.DataFrame({
            "Gene": genes,
            "F_Statistic": f_stats,
            "P_Value": p_values,
            "Adj_P_Value": adjusted_p_values,
            "Max_Mean_Diff": max_mean_diff
        })

        for group in self.groups:
            results[f"Mean_{group}"] = group_means[group]

        results["Significant"] = (
            (results["Adj_P_Value"] < fdr_threshold)
            & (results["Max_Mean_Diff"] >= min_effect_size)
        )

        return results.sort_values("Adj_P_Value").reset_index(drop=True)

    def posthoc(self, gene_name):
        """
        Tukey HSD pairwise comparison for ONE gene, computed on
        demand so it stays instant regardless of dataset size.
        """

        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        gene_row = self.df[self.df[self.gene_column] == gene_name]

        if gene_row.empty:
            return None

        row = gene_row.iloc[0]

        values = np.array([float(row[s]) for s in self.samples])
        labels = np.array([self.group_map[s] for s in self.samples])

        tukey_result = pairwise_tukeyhsd(endog=values, groups=labels)
        summary = tukey_result.summary()

        return pd.DataFrame(
            data=summary.data[1:],
            columns=summary.data[0]
        )

    def gene_expression_table(self, gene_name):
        """Long-format (Sample, Group, Expression) table for one gene — feeds a boxplot."""

        gene_row = self.df[self.df[self.gene_column] == gene_name]

        if gene_row.empty:
            return None

        row = gene_row.iloc[0]

        return pd.DataFrame({
            "Sample": self.samples,
            "Group": [self.group_map[s] for s in self.samples],
            "Expression": [float(row[s]) for s in self.samples]
        })