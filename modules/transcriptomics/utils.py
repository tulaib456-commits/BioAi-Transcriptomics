import pandas as pd


def detect_separator(file_path):

    if file_path.endswith(".csv"):

        return ","

    if file_path.endswith(".tsv"):

        return "\t"

    if file_path.endswith(".txt"):

        return "\t"

    return ","


def read_dataset(file_path):

    separator = detect_separator(file_path)

    if file_path.endswith(".xlsx"):

        return pd.read_excel(file_path)

    return pd.read_csv(

        file_path,

        sep=separator,

        low_memory=False

    )