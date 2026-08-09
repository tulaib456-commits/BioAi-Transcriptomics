import pandas as pd

from modules.transcriptomics.validator import (
    detect_sample_columns
)


class DatasetStatistics:

    def __init__(self, dataframe):

        self.df = dataframe

        self.samples = detect_sample_columns(dataframe)

    def library_sizes(self):

        return self.df[
            self.samples
        ].sum()

    def statistics(self):

        library = self.library_sizes()

        return {

            "Minimum":

                float(library.min()),

            "Maximum":

                float(library.max()),

            "Mean":

                float(library.mean()),

            "Median":

                float(library.median()),

            "Standard Deviation":

                float(library.std())

        }

    def sample_table(self):

        library = self.library_sizes()

        return pd.DataFrame(

            {

                "Sample":

                    library.index,

                "Library Size":

                    library.values

            }

        )