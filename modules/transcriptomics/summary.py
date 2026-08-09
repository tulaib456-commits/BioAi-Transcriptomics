from modules.transcriptomics.validator import (

    detect_gene_column,

    detect_sample_columns,

    count_missing_values,

    duplicate_gene_count

)


def dataset_summary(dataframe):

    genes = detect_gene_column(dataframe)

    samples = detect_sample_columns(dataframe)

    summary = {

        "Rows": dataframe.shape[0],

        "Columns": dataframe.shape[1],

        "Gene Column": genes,

        "Samples": len(samples),

        "Missing Values": count_missing_values(dataframe),

        "Duplicate Genes": duplicate_gene_count(dataframe)

    }

    return summary