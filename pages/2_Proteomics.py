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

require_login()

st.title("🧪 Proteomics")

if "proteomics_dataset" not in st.session_state:
    st.session_state.proteomics_dataset = None
    st.session_state.proteomics_filtered = None
    st.session_state.proteomics_normalized = None

tabs = st.tabs([
    "Upload", "Missing Values", "Normalization",
    "Differential Expression", "Enrichment Analysis", "Machine Learning"
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

                de = DifferentialExpression(data, group_a, group_b)
                de_results = de.run()

                st.session_state.proteomics_de_results = de_results

                st.success(f"{int(de_results['Significant'].sum())} significant proteins")
                display_results_table(de_results, "prot_de")

                plot_factory = PlotFactory(de.df)
                st.plotly_chart(plot_factory.volcano_plot(de_results), use_container_width=True, key="prot_volcano")

        else:

            suggested_map = suggest_multi_groups(samples)
            valid_map = group_assignment_widget(samples, key_prefix="prot_anova", suggested_map=suggested_map)

            if len(set(valid_map.values())) >= 2 and st.button("Run ANOVA", key="prot_anova_run"):

                mgde = MultiGroupDE(data, valid_map, protein_column)
                anova_results = mgde.run()

                st.session_state.proteomics_de_results = anova_results

                st.success(f"{int(anova_results['Significant'].sum())} significant proteins")
                display_results_table(anova_results, "prot_anova")

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

    else:
        st.info("Run Differential Expression first.")