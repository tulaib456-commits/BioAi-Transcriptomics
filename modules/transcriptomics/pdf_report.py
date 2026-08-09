from datetime import datetime
import tempfile
import os

from fpdf import FPDF


class PDFReport(FPDF):

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 60)
        self.cell(0, 10, "BioAI Transcriptomics Report", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, datetime.now().strftime("Generated %Y-%m-%d %H:%M"), ln=True, align="C")
        self.ln(4)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(20, 20, 20)
        self.set_fill_color(235, 235, 245)
        self.cell(0, 8, title, ln=True, fill=True)
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def key_value_table(self, data_dict):
        self.set_font("Helvetica", "", 10)
        for key, value in data_dict.items():
            self.set_font("Helvetica", "B", 10)
            self.cell(70, 7, str(key), border=0)
            self.set_font("Helvetica", "", 10)
            self.cell(0, 7, str(value), ln=True)
        self.ln(2)

    def data_table(self, dataframe, max_rows=20, col_widths=None):

        dataframe = dataframe.head(max_rows)
        columns = list(dataframe.columns)

        if col_widths is None:
            page_width = 190
            col_widths = [page_width / len(columns)] * len(columns)

        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(230, 230, 240)
        for col, width in zip(columns, col_widths):
            self.cell(width, 7, str(col)[:20], border=1, fill=True)
        self.ln()

        self.set_font("Helvetica", "", 8)
        for _, row in dataframe.iterrows():
            for col, width in zip(columns, col_widths):
                value = row[col]
                text = f"{value:.4g}" if isinstance(value, float) else str(value)
                self.cell(width, 6, text[:25], border=1)
            self.ln()
        self.ln(3)

    def add_image(self, image_path, width=170):
        if os.path.exists(image_path):
            self.image(image_path, w=width)
            self.ln(4)


def generate_pdf_report(
    dataset_summary_dict,
    qc_dict,
    filtering_info,
    normalization_method,
    de_results=None,
    de_groups=None,
    anova_results=None,
    enrichment_results=None,
    ml_summary=None,
    dnn_summary=None,
    figures=None
):
    """
    figures: optional dict {title: plotly_figure}. Each is rendered
    to PNG (needs the 'kaleido' package) and embedded. If a figure
    fails to render, its section shows an error note instead of
    breaking the whole report.
    """

    pdf = PDFReport()
    pdf.add_page()

    pdf.section_title("1. Dataset Overview")
    pdf.key_value_table(dataset_summary_dict)

    pdf.section_title("2. Quality Control")
    pdf.key_value_table(qc_dict)

    pdf.section_title("3. Filtering")
    pdf.body_text(filtering_info)

    pdf.section_title("4. Normalization")
    pdf.body_text(f"Method used: {normalization_method}")

    if de_results is not None:
        pdf.add_page()
        pdf.section_title("5. Differential Expression (Two-Group)")
        if de_groups:
            pdf.body_text(
                f"Group A: {len(de_groups[0])} samples | "
                f"Group B: {len(de_groups[1])} samples"
            )
        sig_count = int(de_results["Significant"].sum())
        pdf.body_text(f"{sig_count} significant genes out of {len(de_results)} tested.")
        pdf.data_table(de_results.sort_values("Adj_P_Value").head(20))

    if anova_results is not None:
        pdf.add_page()
        pdf.section_title("6. Multi-Group Differential Expression (ANOVA)")
        sig_count = int(anova_results["Significant"].sum())
        pdf.body_text(f"{sig_count} significant genes out of {len(anova_results)} tested.")
        pdf.data_table(anova_results.sort_values("Adj_P_Value").head(20))

    if enrichment_results is not None:
        pdf.add_page()
        pdf.section_title("7. Enrichment Analysis")
        pdf.data_table(enrichment_results.head(15))

    if ml_summary is not None:
        pdf.add_page()
        pdf.section_title("8. Machine Learning Performance")
        pdf.data_table(ml_summary)

    if dnn_summary is not None:
        pdf.section_title("9. Deep Learning Performance")
        pdf.data_table(dnn_summary)

    if figures:
        pdf.add_page()
        pdf.section_title("10. Key Figures")
        for title, fig in figures.items():
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                fig.write_image(tmp_path, width=900, height=550, scale=2)
                pdf.body_text(title)
                pdf.add_image(tmp_path)
                os.unlink(tmp_path)
            except Exception as error:
                pdf.body_text(f"[Could not render figure '{title}': {error}]")

    return bytes(pdf.output())