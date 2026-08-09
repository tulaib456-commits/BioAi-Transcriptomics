import pandas as pd


class MetadataParser:

    def __init__(self, filepath):

        self.filepath = filepath

        self.metadata = {}

    def parse(self):

        sample_titles = []

        sample_accessions = []

        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as file:

            for line in file:

                if line.startswith("!Sample_title"):

                    sample_titles = line.strip().split("\t")[1:]

                elif line.startswith("!Sample_geo_accession"):

                    sample_accessions = line.strip().split("\t")[1:]

                elif line.startswith("!series_matrix_table_begin"):

                    break

        self.metadata = {

            "Sample Titles": sample_titles,

            "Sample Accessions": sample_accessions,

            "Number of Samples": len(sample_titles)

        }

        return self.metadata