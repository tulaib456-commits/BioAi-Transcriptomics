from modules.transcriptomics.validator import (

    detect_gene_column,

    detect_sample_columns

)


class QualityControl:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

        self.gene = detect_gene_column(self.df)

        self.samples = detect_sample_columns(self.df)

    def basic_statistics(self):

        library_sizes = self.df[
            self.samples
        ].sum()

        return {

            "Genes":

                len(self.df),

            "Samples":

                len(self.samples),

            "Missing":

                int(self.df.isna().sum().sum()),

            "Duplicate":

                int(

                    self.df[self.gene]

                    .duplicated()

                    .sum()

                ),

            "Zero Genes":

                int(

                    (

                        self.df[self.samples]

                        .sum(axis=1)

                        == 0

                    ).sum()

                ),

            "Average Library":

                float(

                    library_sizes.mean()

                ),

            "Minimum Library":

                float(

                    library_sizes.min()

                ),

            "Maximum Library":

                float(

                    library_sizes.max()

                )

        }