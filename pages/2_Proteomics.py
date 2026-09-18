import streamlit as st
import pandas as pd
import numpy as np

from modules.auth import require_login
from modules.proteomics.missing_value_handler import (
    MissingValueAnalyzer, ProteomicsFilter, Imputer
)
from modules.proteomics.normalization import ProteomicsNormalizer

# Reused directly from the Transcriptomics engine -- same
# statistical logic applies once data is protein x sample shaped
from modules.transcriptomics.validator import detect_gene_column, detect_sample_columns
from modules.transcriptomics.gene_id_mapper import detect_id_type, GeneIDConverter
from modules.transcriptomics.differential_expression import DifferentialExpression, suggest_groups
from modules.transcriptomics.multi_group_de import MultiGroupDE, suggest_multi_groups
from modules.transcriptomics.group_assignment import group_assignment_widget
from modules.transcriptomics.enrichment import EnrichmentAnalysis, ENRICHR_LIBRARIES, SUPPORTED_ORGANISMS
from modules.transcriptomics.feature_builder import FeatureBuilder
from modules.transcriptomics.ml_classifier import MLClassifier
from modules.transcriptomics.dnn_classifier import DNNClassifier
from modules.transcriptomics.model_interpretation import ROCAnalysis, ModelInterpreter
from modules.transcriptomics.plots import PlotFactory
from modules.transcriptomics.result_display import display_results_table
from modules.transcriptomics.pdf_report import generate_pdf_report

require_login()

st.title("🧪 Proteomics")

if "proteomics_dataset" not in st.session_state:
    st.session_state.proteomics_dataset = None
    st.session_state.proteomics_filtered = None
    st.session_state.proteomics_normalized = None

tabs = st.tabs([
    "Upload", "Missing Values", "Normalization",
    "Differential Expression", "Enrichment Analysis", "Machine Learning", "Export"
])

with tabs[0]:

    st.header("Upload Protein Intensity Matrix")
    st.caption(
        "Rows = proteins (UniProt ID or gene symbol), columns = "
        "samples, values = intensity. Same format as your "
        "Transcriptomics count matrix."
    )

    uploaded = st.file_uploader("Upload dataset", type=["csv", "tsv", "txt", "xlsx"])

    if uploaded is not None:

        if uploaded.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded)
        else:
            sep = "\t" if uploaded.name.endswith((".tsv", ".txt")) else ","
            data = pd.read_csv(uploaded, sep=sep)

        st.session_state.proteomics_dataset = data
        st.success(f"Uploaded: {len(data)} proteins, {len(data.columns) - 1} samples")
        st.dataframe(data.head())

with tabs[1]:

    if st.session_state.proteomics_dataset is not None:

        st.header("Missing Value Analysis & Imputation")

        protein_column = detect_gene_column(st.session_state.proteomics_dataset)
        samples = detect_sample_columns(st.session_state.proteomics_dataset)

        analyzer = MissingValueAnalyzer(
            st.session_state.proteomics_dataset, protein_column, samples
        )

        summary = analyzer.missingness_summary()

        st.metric(
            "Overall missing values",
            f"{summary['Missing_Fraction'].mean() * 100:.1f}%"
        )

        st.caption(
            "Assign groups below to filter/impute correctly -- a "
            "protein detected in one group but not another is "
            "meaningful signal, not noise."
        )

        group_map = group_assignment_widget(samples, key_prefix="proteomics_mv")

        if len(set(group_map.values())) >= 2:

            min_valid = st.slider(
                "Minimum valid-value fraction (in at least one group)",
                0.3, 1.0, 0.7
            )

            if st.button("Filter and Impute"):

                prot_filter = ProteomicsFilter(
                    st.session_state.proteomics_dataset, protein_column, samples
                )
                filtered = prot_filter.filter_by_group_detection(group_map, min_valid)

                imputed = Imputer().impute(filtered, samples)

                st.session_state.proteomics_filtered = imputed

                st.success(
                    f"{len(filtered)} of {len(st.session_state.proteomics_dataset)} "
                    "proteins retained and imputed."
                )

        if st.session_state.proteomics_filtered is not None:
            st.dataframe(st.session_state.proteomics_filtered.head())

    else:
        st.info("Upload a dataset first.")

with tabs[2]:

    if st.session_state.proteomics_filtered is not None:

        st.header("Normalization")

        method = st.selectbox(
            "Normalization Method",
            ["Median Normalization (recommended)", "Quantile Normalization"]
        )

        samples = detect_sample_columns(st.session_state.proteomics_filtered)
        normalizer = ProteomicsNormalizer(st.session_state.proteomics_filtered, samples)

        if st.button("Apply Normalization"):

            if method.startswith("Median"):
                st.session_state.proteomics_normalized = normalizer.median_normalization()
            else:
                st.session_state.proteomics_normalized = normalizer.quantile_normalization()

        if st.session_state.proteomics_normalized is not None:
            st.dataframe(st.session_state.proteomics_normalized.head())

    else:
        st.info("Complete Missing Value handling first.")

with tabs[3]:

    if st.session_state.proteomics_normalized is not None:

        st.header("Differential Expression")
        st.caption(
            "Same engine as Transcriptomics -- proven, tested logic, "
            "applied to protein abundance instead of gene expression."
        )

        data = st.session_state.proteomics_normalized
        protein_column = detect_gene_column(data)
        samples = detect_sample_columns(data)

        de_mode = st.radio(
            "Comparison Type",
            ["Two-Group (t-test)", "Multi-Group (ANOVA, 3+ conditions)"],
            horizontal=True, key="prot_de_mode"
        )

        if de_mode == "Two-Group (t-test)":

            suggested_a, suggested_b = suggest_groups(samples)

            col1, col2 = st.columns(2)
            with col1:
                group_a = st.multiselect("Group A", samples, default=suggested_a or [], key="prot_group_a")
            with col2:
                group_b = st.multiselect(
                    "Group B", [s for s in samples if s not in group_a],
                    default=[s for s in (suggested_b or []) if s not in group_a],
                    key="prot_group_b"
                )

            if group_a and group_b:

                de = DifferentialExpression(data, group_a, group_b, already_normalized=True)
                de_results = de.run()

                st.session_state.proteomics_de_results = de_results

                st.success(f"{int(de_results['Significant'].sum())} significant proteins")
                display_results_table(de_results, "prot_de")

                plot_factory = PlotFactory(de.df)
                st.plotly_chart(plot_factory.volcano_plot(de_results), use_container_width=True, key="prot_volcano")

                st.subheader("PCA")
                pca_group_map = {s: "Group A" for s in group_a}
                pca_group_map.update({s: "Group B" for s in group_b})
                st.plotly_chart(plot_factory.pca_plot(pca_group_map), use_container_width=True, key="prot_pca")

        else:

            suggested_map = suggest_multi_groups(samples)
            valid_map = group_assignment_widget(samples, key_prefix="prot_anova", suggested_map=suggested_map)

            if len(set(valid_map.values())) >= 2 and st.button("Run ANOVA", key="prot_anova_run"):

                mgde = MultiGroupDE(data, valid_map, protein_column, already_normalized=True)
                anova_results = mgde.run()

                st.session_state.proteomics_de_results = anova_results

                st.success(f"{int(anova_results['Significant'].sum())} significant proteins")
                display_results_table(anova_results, "prot_anova")
                
                st.subheader("Inspect a Protein")

                top_proteins = anova_results.head(30)[anova_results.columns[0]].tolist()

                selected_protein = st.selectbox(
                    "Choose a significant protein to inspect",
                    top_proteins, key="prot_anova_select"
                )

                plot_factory = PlotFactory(data)

                st.subheader("Compare Multiple Proteins")

                multi_proteins = st.multiselect(
                    "Select proteins to compare (heatmap)",
                    top_proteins, default=top_proteins[:10], key="prot_anova_multi"
                )

                if multi_proteins:

                    expr_rows = []
                    for protein in multi_proteins:
                        row = mgde.df[mgde.df[protein_column] == protein]
                        if not row.empty:
                            expr_rows.append(row[mgde.samples].iloc[0])

                    if expr_rows:
                        expr_matrix = pd.DataFrame(expr_rows, index=multi_proteins)
                        st.plotly_chart(
                            plot_factory.top_genes_heatmap(expr_matrix),
                            use_container_width=True, key="prot_anova_heatmap"
                        )

                if selected_protein:

                    expression_table = mgde.gene_expression_table(selected_protein)

                    st.plotly_chart(
                        plot_factory.gene_boxplot(expression_table, selected_protein),
                        use_container_width=True, key="prot_anova_box"
                    )

                    st.caption("Tukey HSD pairwise comparison:")
                    posthoc_table = mgde.posthoc(selected_protein)
                    st.dataframe(posthoc_table)
    else:
        st.info("Complete Normalization first.")

with tabs[4]:

    if st.session_state.get("proteomics_de_results") is not None:

        st.header("Enrichment Analysis")

        de_results = st.session_state.proteomics_de_results
        significant = de_results.loc[de_results["Significant"], de_results.columns[0]].tolist()

        st.caption(f"{len(significant)} significant proteins available.")

        library_label = st.selectbox("Gene Set Library", list(ENRICHR_LIBRARIES.keys()), key="prot_enrich_lib")
        organism = st.selectbox("Organism", SUPPORTED_ORGANISMS, key="prot_enrich_org")

        if st.button("Run Enrichment Analysis", key="prot_enrich_run"):

            enrichment = EnrichmentAnalysis(significant)
            results, error = enrichment.run(library_label, organism=organism)

            if error:
                st.warning(error)
            else:
                st.session_state.proteomics_enrichment = results
                display_results_table(results, "prot_enrichment", long_text_columns=["Matched_Genes", "Pathway"])

                plot_factory = PlotFactory(st.session_state.proteomics_normalized)
                st.plotly_chart(plot_factory.enrichment_bar_plot(results), use_container_width=True, key="prot_enrich_bar")

    else:
        st.info("Run Differential Expression first.")

with tabs[5]:

    if st.session_state.get("proteomics_de_results") is not None:

        st.header("Machine Learning")
        st.caption("Same cross-validated engine as Transcriptomics.")

        data = st.session_state.proteomics_normalized
        protein_column = detect_gene_column(data)
        samples = detect_sample_columns(data)

        de_results = st.session_state.proteomics_de_results
        significant_proteins = de_results.loc[de_results["Significant"], de_results.columns[0]].tolist()

        group_map = group_assignment_widget(samples, key_prefix="prot_ml")

        if len(set(group_map.values())) >= 2 and significant_proteins:

            fb = FeatureBuilder(data, protein_column)
            X, y, features, nan_count = fb.build(group_map, gene_list=significant_proteins)

            ml = MLClassifier(X, y)

            if st.button("Run Cross-Validation", key="prot_ml_run"):

                summary, detail = ml.evaluate(list(ml.models.keys()))
                st.session_state.prot_ml_summary = summary

            if st.session_state.get("prot_ml_summary") is not None:
                st.dataframe(st.session_state.prot_ml_summary)

                best_model = st.session_state.prot_ml_summary.iloc[0]["Model"]

                cm = ml.confusion_matrix_cv(best_model)
                plot_factory = PlotFactory(data)
                st.plotly_chart(plot_factory.confusion_matrix_heatmap(cm), use_container_width=True, key="prot_cm")

                roc_analysis = ROCAnalysis(ml, best_model)
                curves = roc_analysis.curves()

                col_roc, col_pr = st.columns(2)
                with col_roc:
                    st.plotly_chart(plot_factory.roc_curve_plot(curves), use_container_width=True, key="prot_roc")
                with col_pr:
                    st.plotly_chart(plot_factory.pr_curve_plot(curves), use_container_width=True, key="prot_pr")

                if st.button("Compute SHAP Importance", key="prot_shap_btn"):
                    final_pipeline = ml.fit_final_model(best_model)
                    interpreter = ModelInterpreter(final_pipeline, ml.X)
                    shap_values, shap_error = interpreter.compute()

                    if shap_error:
                        st.warning(shap_error)
                    else:
                        importance = interpreter.importance_table(shap_values)
                        st.dataframe(importance.head(20))
                        st.plotly_chart(plot_factory.shap_importance_plot(importance), use_container_width=True, key="prot_shap")

    else:
        st.info("Run Differential Expression first.")


with tabs[6]:

    st.header("Export")

    if st.session_state.proteomics_dataset is not None:
        st.download_button(
            "Download Raw Dataset (CSV)",
            st.session_state.proteomics_dataset.to_csv(index=False),
            file_name="proteomics_raw.csv", mime="text/csv"
        )

    if st.session_state.proteomics_filtered is not None:
        st.download_button(
            "Download Filtered/Imputed Dataset (CSV)",
            st.session_state.proteomics_filtered.to_csv(index=False),
            file_name="proteomics_filtered.csv", mime="text/csv"
        )

    if st.session_state.proteomics_normalized is not None:
        st.download_button(
            "Download Normalized Dataset (CSV)",
            st.session_state.proteomics_normalized.to_csv(index=False),
            file_name="proteomics_normalized.csv", mime="text/csv"
        )

    if st.session_state.get("proteomics_de_results") is not None:
        st.download_button(
            "Download Differential Expression Results (CSV)",
            st.session_state.proteomics_de_results.to_csv(index=False),
            file_name="proteomics_de_results.csv", mime="text/csv"
        )

    if st.session_state.get("proteomics_enrichment") is not None:
        st.download_button(
            "Download Enrichment Results (CSV)",
            st.session_state.proteomics_enrichment.to_csv(index=False),
            file_name="proteomics_enrichment.csv", mime="text/csv"
        )

        st.divider()
        st.subheader("Full PDF Report")

        if st.button("Generate PDF Report", key="prot_pdf_btn"):

            figures = {}
            if st.session_state.get("proteomics_enrichment") is not None:
                pf = PlotFactory(st.session_state.proteomics_normalized)
                figures["Top Enriched Pathways"] = pf.enrichment_bar_plot(st.session_state.proteomics_enrichment)

            pdf_bytes = generate_pdf_report(
                dataset_summary_dict={"Proteins": len(st.session_state.proteomics_dataset), "Samples": len(detect_sample_columns(st.session_state.proteomics_dataset))},
                qc_dict={"Missing values handled": "Yes" if st.session_state.proteomics_filtered is not None else "No"},
                filtering_info=f"{len(st.session_state.proteomics_filtered)} proteins after missing-value filtering." if st.session_state.proteomics_filtered is not None else "Not yet run.",
                normalization_method="Applied" if st.session_state.proteomics_normalized is not None else "Not yet run",
                de_results=st.session_state.get("proteomics_de_results"),
                enrichment_results=st.session_state.get("proteomics_enrichment"),
                ml_summary=st.session_state.get("prot_ml_summary"),
                figures=figures if figures else None
            )

            st.session_state.prot_pdf_bytes = pdf_bytes

        if st.session_state.get("prot_pdf_bytes") is not None:
            st.download_button("Download PDF Report", st.session_state.prot_pdf_bytes, file_name="Proteomics_Report.pdf", mime="application/pdf", key="prot_pdf_download")