from datetime import datetime


class AnalysisTracker:

    def __init__(self):

        self.history = []

    def log(self, action):

        self.history.append(

            {

                "Time":

                    datetime.now(),

                "Action":

                    action

            }

        )

    def get_history(self):

        return self.history