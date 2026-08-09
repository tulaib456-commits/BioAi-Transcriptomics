import os

from datetime import datetime


class BioAILogger:

    def __init__(self):

        os.makedirs("logs", exist_ok=True)

        self.logfile = "logs/bioai.log"

    def write(self, message):

        with open(self.logfile, "a") as file:

            file.write(

                f"{datetime.now()} : {message}\n"

            )