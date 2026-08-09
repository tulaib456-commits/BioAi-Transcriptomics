import os

from config.settings import SUPPORTED_FILE_TYPES


def validate_file_type(filename):

    extension = os.path.splitext(filename)[1]

    return extension.lower() in SUPPORTED_FILE_TYPES