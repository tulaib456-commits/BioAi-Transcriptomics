from modules.transcriptomics.utils import read_dataset


def load_transcriptomics_dataset(file):

    dataframe = read_dataset(file)

    return dataframe