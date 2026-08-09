import os
import shutil


class DatasetStorage:

    def __init__(self):

        self.upload_folder = "uploads"

        os.makedirs(self.upload_folder, exist_ok=True)

    def save(self, source_path):

        filename = os.path.basename(source_path)

        destination = os.path.join(
            self.upload_folder,
            filename
        )

        shutil.copy(source_path, destination)

        return destination