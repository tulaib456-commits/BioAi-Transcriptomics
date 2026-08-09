import os

from modules.transcriptomics.validator import (

    detect_sample_columns,

    detect_gene_column

)


class ProjectInformation:

    def __init__(self, dataframe, filename):

        self.df = dataframe

        self.filename = filename

    def summary(self):

        size = round(

            os.path.getsize(self.filename) / 1024,

            2

        )

        return {

            "Dataset":

                os.path.basename(self.filename),

            "Genes":

                len(self.df),

            "Samples":

                len(

                    detect_sample_columns(

                        self.df

                    )

                ),

            "Gene Column":

                detect_gene_column(

                    self.df

                ),

            "File Size (KB)":

                size

        }