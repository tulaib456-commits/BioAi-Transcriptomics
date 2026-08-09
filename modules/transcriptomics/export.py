import os


class Exporter:

    def __init__(self):

        os.makedirs("exports", exist_ok=True)

    def save(self, dataframe, filename):

        path = os.path.join("exports", filename)

        dataframe.to_csv(path, index=False)

        return path