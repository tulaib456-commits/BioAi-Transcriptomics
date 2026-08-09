import pandas as pd

from config.constants import GENE_COLUMN_CANDIDATES


def detect_gene_column(dataframe):

    for column in dataframe.columns:

        if column in GENE_COLUMN_CANDIDATES:

            return column

    return dataframe.columns[0]


def detect_sample_columns(dataframe):

    gene_column = detect_gene_column(dataframe)

    return [

        column

        for column in dataframe.columns

        if column != gene_column

    ]


def count_missing_values(dataframe):

    return dataframe.isna().sum().sum()


def duplicate_gene_count(dataframe):

    gene_column = detect_gene_column(dataframe)

    return dataframe[gene_column].duplicated().sum()