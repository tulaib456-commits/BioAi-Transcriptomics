import numpy as np
import pandas as pd


class FeatureBuilder:
    """
    Converts a genes-as-rows / samples-as-columns expression matrix
    into a samples-as-rows / genes-as-columns feature matrix with a
    label column — the format every ML library (scikit-learn, Keras)
    actually expects.
    """

    def __init__(self, dataframe, gene_column):
        self.dataframe = dataframe
        self.gene_column = gene_column

    def build(self, group_map, gene_list=None, normalize=True):
        """
        group_map : dict {sample: group_label}
        gene_list : list of gene names to use as features.
                    If None, ALL genes are used (not recommended —
                    with small sample counts this badly overfits).
        normalize : applies CPM + log2 internally if True.

        Returns (X, y, feature_names, nan_count):
          X : DataFrame, shape (n_samples, n_genes)
          y : Series of group labels, aligned to X's index
          feature_names : list of gene names actually used
          nan_count : how many missing values were found & filled with 0
        """

        samples = list(group_map.keys())

        missing_samples = [
            s for s in samples if s not in self.dataframe.columns
        ]

        if missing_samples:
            raise ValueError(
                "These samples are in the group assignment but not "
                f"in the dataset: {missing_samples}"
            )

        working = self.dataframe.copy()

        if normalize:
            totals = working[samples].sum()
            working[samples] = (working[samples] / totals) * 1_000_000
            working[samples] = np.log2(working[samples] + 1)

        if gene_list is not None:
            working = working[
                working[self.gene_column].isin(gene_list)
            ]

        working = working.drop_duplicates(subset=self.gene_column)

        feature_names = working[self.gene_column].tolist()

        X = working[samples].T
        X.columns = feature_names
        X.index = samples

        nan_count = int(X.isna().sum().sum())
        if nan_count > 0:
            X = X.fillna(0)

        y = pd.Series(
            [group_map[s] for s in samples],
            index=samples,
            name="Label"
        )

        return X, y, feature_names, nan_count