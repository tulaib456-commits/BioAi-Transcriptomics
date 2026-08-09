class ReportGenerator:

    def __init__(self, summary, qc):

        self.summary = summary

        self.qc = qc

    def generate(self):

        report = []

        report.append("BIOAI DATASET REPORT")

        report.append("=" * 40)

        report.append("")

        report.append("DATASET SUMMARY")

        for key, value in self.summary.items():

            report.append(f"{key}: {value}")

        report.append("")

        report.append("QUALITY CONTROL")

        for key, value in self.qc.items():

            report.append(f"{key}: {value}")

        return "\n".join(report)