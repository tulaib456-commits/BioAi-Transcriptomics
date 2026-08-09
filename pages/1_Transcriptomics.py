import pandas as pd
import numpy as np

from modules.transcriptomics.pdf_report import generate_pdf_report
import base64
import streamlit.components.v1 as components
from modules.transcriptomics.result_display import display_results_table
from modules.transcriptomics.group_assignment import group_assignment_widget
from modules.transcriptomics.feature_builder import FeatureBuilder
from modules.transcriptomics.ml_classifier import MLClassifier
from modules.transcriptomics.qc_diagnostics import QCDiagnostics
from modules.transcriptomics.multi_group_de import (
    MultiGroupDE,
    suggest_multi_groups
)
from modules.transcriptomics.model_interpretation import (
    ROCAnalysis,
    ModelInterpreter
)
from modules.transcriptomics.gene_id_mapper import (
    detect_id_type,
    GeneIDConverter
)
from modules.transcriptomics.dnn_classifier import DNNClassifier
from modules.transcriptomics.model_registry import (
    save_classical_model,
    save_dnn_model,
    list_saved_models
)

from modules.transcriptomics.validator import detect_gene_column, detect_sample_columns
from modules.transcriptomics.enrichment import (
    EnrichmentAnalysis,
    ENRICHR_LIBRARIES,
    SUPPORTED_ORGANISMS
)
import os
import tempfile

import streamlit as st

from modules.transcriptomics.loader import load_transcriptomics_dataset
from modules.transcriptomics.summary import dataset_summary
from modules.transcriptomics.metadata import MetadataParser
from modules.transcriptomics.storage import DatasetStorage
from modules.transcriptomics.logger import BioAILogger

from modules.transcriptomics.dashboard import Dashboard
from modules.transcriptomics.qc import QualityControl

from modules.transcriptomics.export import Exporter
from modules.transcriptomics.report import ReportGenerator

from modules.transcriptomics.filters import GeneFilter
from modules.transcriptomics.statistics import DatasetStatistics
from modules.transcriptomics.normalization import Normalizer
from modules.transcriptomics.session import SessionManager

from modules.transcriptomics.differential_expression import (
    DifferentialExpression,
    suggest_groups
)
from modules.transcriptomics.plots import PlotFactory
from modules.transcriptomics.validator import detect_sample_columns


st.set_page_config(
    page_title="Transcriptomics",
    page_icon="🧬",
    layout="wide"
)

SessionManager.initialize()

st.title("🧬 BioAI Transcriptomics")

tabs = st.tabs(
    [
        "Upload",
        "Summary",
        "Quality Control",
        "Filtering",
        "Normalization",
        "Differential Expression",
        "Enrichment Analysis",
        "Machine Learning",
        "Export"
    ]
)
with tabs[0]:

    st.header("Upload Dataset")

    dataset = st.file_uploader(
        "RNA Seq Count Matrix",
        type=["csv", "tsv", "txt", "xlsx"]
    )

    metadata = st.file_uploader(
        "GEO Series Matrix",
        type=["txt"]
    )

    if dataset:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(dataset.name)[1]
        ) as tmp:

            tmp.write(dataset.getbuffer())

            dataset_path = tmp.name

        storage = DatasetStorage()

        logger = BioAILogger()

        saved_dataset = storage.save(dataset_path)

        dataframe = load_transcriptomics_dataset(
            saved_dataset
        )

        st.session_state.dataset = dataframe

        logger.write("Dataset Uploaded")

        if metadata:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".txt"
            ) as tmp:

                tmp.write(metadata.getbuffer())

                metadata_path = tmp.name

            saved_metadata = storage.save(
                metadata_path
            )

            parser = MetadataParser(
                saved_metadata
            )

            st.session_state.metadata = parser.parse()

            logger.write("Metadata Uploaded")

        st.success("Upload Successful")
with tabs[1]:

    if st.session_state.dataset is not None:
        
        st.header("Dataset Summary")

        summary = dataset_summary(
            st.session_state.dataset
        )

        Dashboard.dataset_metrics(summary)

        st.dataframe(
            st.session_state.dataset.head()
        )

        stats = DatasetStatistics(
            st.session_state.dataset
        )

        st.subheader("Library Statistics")

        st.json(
            stats.statistics()
        )

        st.subheader("Sample Information")

        st.dataframe(
            stats.sample_table()
        )
with tabs[2]:

    if st.session_state.dataset is not None:

        st.header("Quality Control")

        diagnostics = QCDiagnostics(st.session_state.dataset)
        plot_factory = PlotFactory(st.session_state.dataset)

        st.subheader("Library Sizes")
        st.plotly_chart(
            plot_factory.library_size_bar(diagnostics.library_sizes()),
            use_container_width=True, key="qc_library_size"
        )

        st.divider()
        st.subheader("Sample-Sample Correlation")
        st.caption(
            "Samples from the same group should correlate highly. "
            "A sample that stands out here may be mislabeled, "
            "degraded, or a batch outlier."
        )

        if st.button("Compute Sample Correlation"):
            with st.spinner("Computing correlations..."):
                st.session_state.qc_corr_matrix = diagnostics.correlation_matrix()

        if st.session_state.get("qc_corr_matrix") is not None:

            corr_matrix = st.session_state.qc_corr_matrix

            st.plotly_chart(
                plot_factory.correlation_heatmap(corr_matrix),
                use_container_width=True, key="qc_correlation"
            )

            mean_corr, flagged = diagnostics.flag_outlier_samples()

            if flagged:
                st.warning(
                    f"Potential outlier sample(s), based on low average "
                    f"correlation to others: {', '.join(flagged)}."
                )
            else:
                st.success("No samples flagged as correlation outliers.")

        st.divider()
        st.subheader("Expression Distribution")

        if st.button("Show Expression Distribution"):
            st.session_state.qc_density_fig = plot_factory.expression_density(
                diagnostics.log_cpm()
            )

        if st.session_state.get("qc_density_fig") is not None:
            st.plotly_chart(
                st.session_state.qc_density_fig,
                use_container_width=True, key="qc_density"
            )

        st.divider()
        st.subheader("Top Expressed Genes")
        st.dataframe(diagnostics.top_expressed_genes(15))

    else:
        st.info("Upload a dataset first.")
with tabs[3]:

    if st.session_state.dataset is not None:

        st.header("Gene Filtering")

        filter_method = st.radio(
            "Choose filtering approach",
            [
                "Raw count threshold (simple)",
                "CPM-based threshold (recommended — adjusts for library size)"
            ],
            index=1
        )

        if filter_method.startswith("CPM"):

            min_cpm = st.slider("Minimum CPM", 0.1, 5.0, 1.0)

            cpm_samples = detect_sample_columns(st.session_state.dataset)

            min_samples_cpm = st.slider(
                "Minimum samples meeting CPM threshold", 1, len(cpm_samples), 2,
                key="cpm_min_samples"
            )

        else:

            minimum = st.slider("Minimum Total Count", 1, 100, 10)

            raw_samples = detect_sample_columns(st.session_state.dataset)

            min_samples_raw = st.slider(
                "Minimum Samples", 1, len(raw_samples), 2, key="raw_min_samples"
            )

        if st.button("Apply Filter"):

            if filter_method.startswith("CPM"):

                gene_filter = GeneFilter(st.session_state.dataset)
                filtered = gene_filter.filter_by_cpm(
                    min_cpm=min_cpm, min_samples=min_samples_cpm
                )

            else:

                filter_engine = GeneFilter(st.session_state.dataset)
                filtered = filter_engine.remove_zero_genes()

                filter_engine = GeneFilter(filtered)
                filtered = filter_engine.remove_low_count_genes(minimum)

                filter_engine = GeneFilter(filtered)
                filtered = filter_engine.remove_sparse_genes(min_samples_raw)

            st.session_state.filtered = filtered

        if st.session_state.filtered is not None:

            st.info(
                f"Before filtering: {len(st.session_state.dataset)} genes → "
                f"After: {len(st.session_state.filtered)} genes "
                f"({len(st.session_state.dataset) - len(st.session_state.filtered)} removed)"
            )

            st.divider()

            gene_column = detect_gene_column(st.session_state.dataset)
            id_type = detect_id_type(st.session_state.dataset[gene_column])

            if id_type in ("ensembl", "entrez"):

                st.warning(
                    f"Your gene IDs look like **{id_type.upper()}** IDs, "
                    "not gene symbols (e.g. TNFAIP3). Enrichment Analysis "
                    "and gene-name readability need symbols. Convert now."
                )

                st.caption(
                    "Upload NCBI's Human.GRCh38.p13.annot.tsv(.gz) file "
                    "(from the same GEO download page) for an instant, "
                    "offline conversion."
                )

                annotation_file = st.file_uploader(
                    "Upload NCBI annotation file",
                    type=["tsv", "gz"], key="annotation_upload"
                )

                if annotation_file is not None and st.button("Convert Using Annotation File"):

                    converter = GeneIDConverter(st.session_state.filtered, gene_column)

                    converted, stats, error = converter.convert_using_annotation_file(
                        annotation_file
                    )

                    if error:
                        st.error(error)
                    else:
                        st.session_state.filtered = converted

                        st.session_state.normalized = None
                        st.session_state.de_results = None
                        st.session_state.de_groups = None
                        st.session_state.multi_de_results = None
                        st.session_state.multi_de_engine = None
                        st.session_state.enrichment_results = None
                        st.session_state.ml_summary = None
                        st.session_state.ml_engine = None
                        st.session_state.dnn_summary = None
                        st.session_state.dnn_cm = None
                        st.session_state.dnn_engine = None

                        st.success(
                            f"Converted: {stats['Successfully Mapped']} of "
                            f"{stats['Total IDs']} IDs mapped, "
                            f"{stats['Final Gene Count (after merging duplicates)']} final genes. "
                            "Previous Normalization/DE/ANOVA/ML results were "
                            "cleared — please re-run those steps."
                        )
                        st.rerun()

            st.dataframe(st.session_state.filtered.head())

    else:
        st.info("Upload a dataset first.")
with tabs[4]:

    if st.session_state.filtered is not None:

        st.header("Normalization")

        normalization_method = st.selectbox(
            "Normalization Method",
            ["CPM", "Log2 CPM", "Median-of-Ratios (DESeq2-style, recommended)"]
        )

        if st.button("Apply Normalization"):

            normalizer = Normalizer(st.session_state.filtered)

            if normalization_method == "CPM":
                normalized = normalizer.cpm()
            elif normalization_method == "Log2 CPM":
                normalized = normalizer.log2_cpm()
            else:
                normalized = normalizer.median_of_ratios()

            st.session_state.normalized = normalized

        if st.session_state.normalized is not None:

            st.dataframe(st.session_state.normalized.head())

            st.divider()
            st.subheader("Normalization Diagnostics")
            st.caption(
                "Should look more even/centered than the same plot "
                "on raw, unnormalized data — visual proof it worked."
            )

            if st.button("Show RLE Diagnostic Plot"):

                gene_column = detect_gene_column(st.session_state.filtered)
                samples = detect_sample_columns(st.session_state.filtered)

                totals = st.session_state.filtered[samples].sum()
                log_cpm_check = np.log2(
                    (st.session_state.filtered[samples] / totals) * 1_000_000 + 1
                )

                plot_factory = PlotFactory(st.session_state.filtered)

                st.session_state.norm_rle_fig = plot_factory.rle_plot(log_cpm_check)

            if st.session_state.get("norm_rle_fig") is not None:
                st.plotly_chart(
                    st.session_state.norm_rle_fig,
                    use_container_width=True, key="norm_rle"
                )

    else:
        st.info("Run Filtering first before Normalization.")
with tabs[5]:

    if st.session_state.filtered is not None:

        st.header("Differential Expression")

        samples = detect_sample_columns(
            st.session_state.filtered
        )

        de_mode = st.radio(
            "Comparison Type",
            ["Two-Group (t-test)", "Multi-Group (ANOVA, 3+ conditions)"],
            horizontal=True
        )

        if de_mode == "Two-Group (t-test)":

            suggested_a, suggested_b = suggest_groups(samples)

            st.caption(
                "Assign samples to two groups to compare "
                "(e.g. control vs treated)."
            )

            col1, col2 = st.columns(2)

            with col1:

                group_a = st.multiselect(
                    "Group A (baseline)",
                    samples,
                    default=suggested_a if suggested_a else []
                )

            with col2:

                group_b = st.multiselect(
                    "Group B (comparison)",
                    [s for s in samples if s not in group_a],
                    default=(
                        [s for s in suggested_b if s not in group_a]
                        if suggested_b else []
                    )
                )

            fdr_threshold = st.slider(
                "Adjusted P-Value Threshold",
                0.01, 0.20, 0.05
            )

            log2fc_threshold = st.slider(
                "Log2 Fold Change Threshold",
                0.0, 5.0, 1.0
            )

            if group_a and group_b:

                de = DifferentialExpression(
                    st.session_state.filtered,
                    group_a,
                    group_b
                )

                de_results = de.run(
                    fdr_threshold=fdr_threshold,
                    log2fc_threshold=log2fc_threshold
                )

                st.session_state.de_results = de_results
                st.session_state.de_groups = (group_a, group_b)

                significant_count = int(
                    de_results["Significant"].sum()
                )

                st.success(
                    f"{significant_count} significant genes found "
                    f"out of {len(de_results)}"
                )

                display_results_table(de_results, "de_two_group")

                st.subheader("Volcano Plot")

                plot_factory = PlotFactory(de.df)

                st.plotly_chart(
                    plot_factory.volcano_plot(
                        de_results,
                        log2fc_threshold=log2fc_threshold
                    ),
                    use_container_width=True
                )

                st.subheader("PCA")

                group_map = {
                    sample: "Group A" for sample in group_a
                }
                group_map.update(
                    {sample: "Group B" for sample in group_b}
                )

                st.plotly_chart(
                    plot_factory.pca_plot(group_map),
                    use_container_width=True
                )

            else:

                st.info(
                    "Select samples for both Group A and Group B "
                    "to run differential expression."
                )

        elif de_mode == "Multi-Group (ANOVA, 3+ conditions)":

            gene_column = detect_gene_column(st.session_state.filtered)
            suggested_map = suggest_multi_groups(samples)

            valid_map = group_assignment_widget(
                samples, key_prefix="anova", suggested_map=suggested_map
            )

            distinct_groups = set(valid_map.values())
            group_counts = pd.Series(list(valid_map.values())).value_counts()

            if len(valid_map) < len(samples):
                st.warning(
                    f"{len(samples) - len(valid_map)} sample(s) have "
                    "no group assigned and will be excluded."
                )

            if len(distinct_groups) < 2:
                st.info("Assign at least 2 distinct groups to run ANOVA.")
            else:

                if (group_counts < 2).any():
                    st.warning(
                        "Some groups have only 1 sample — results for "
                        "those will be less statistically reliable."
                    )

                anova_fdr = st.slider(
                    "ANOVA Adjusted P-Value Threshold",
                    0.01, 0.20, 0.05,
                    key="anova_fdr"
                )
                anova_effect_size = st.slider(
                    "Minimum Effect Size (max mean difference between groups, log2 scale)",
                    0.0, 5.0, 1.0,
                    key="anova_effect"
                )

                if st.button("Run Multi-Group ANOVA"):

                    mgde = MultiGroupDE(
                        st.session_state.filtered,
                        valid_map,
                        gene_column
                    )

                    anova_results = mgde.run(
                        fdr_threshold=anova_fdr,
                        min_effect_size=anova_effect_size
                    )

                    st.session_state.multi_de_results = anova_results
                    st.session_state.multi_de_engine = mgde

                if st.session_state.get("multi_de_results") is not None:

                    anova_results = st.session_state.multi_de_results
                    mgde = st.session_state.multi_de_engine

                    sig_count = int(anova_results["Significant"].sum())

                    st.success(
                        f"{sig_count} significant genes out of "
                        f"{len(anova_results)}"
                    )

                    display_results_table(anova_results, "anova")

                    st.subheader("Inspect a Gene")

                    top_genes = anova_results.head(30)["Gene"].tolist()

                    selected_gene = st.selectbox(
                        "Choose a significant gene to inspect",
                        top_genes
                    )
                    st.subheader("Compare Multiple Genes")
                    plot_factory = PlotFactory(st.session_state.filtered)
                    multi_genes = st.multiselect(
                        "Select genes to compare (heatmap)",
                        top_genes,
                        default=top_genes[:10]
                    )

                    if multi_genes:

                        expr_rows = []
                        for gene in multi_genes:
                            row = mgde.df[mgde.df[gene_column] == gene]
                            if not row.empty:
                                expr_rows.append(row[mgde.samples].iloc[0])

                        if expr_rows:
                            expr_matrix = pd.DataFrame(expr_rows, index=multi_genes)
                            st.plotly_chart(
                                plot_factory.top_genes_heatmap(expr_matrix),
                                use_container_width=True,
                                key="anova_heatmap"
                            )

                    if selected_gene:

                        expression_table = mgde.gene_expression_table(
                            selected_gene
                        )

                        plot_factory = PlotFactory(st.session_state.filtered)

                        st.plotly_chart(
                            plot_factory.gene_boxplot(
                                expression_table, selected_gene
                            ),
                            use_container_width=True
                        )

                        st.caption("Tukey HSD pairwise comparison:")

                        posthoc_table = mgde.posthoc(selected_gene)

                        st.dataframe(posthoc_table)

    else:

        st.info(
            "Run Filtering first before Differential Expression."
        )
with tabs[6]:

    st.header("Enrichment Analysis")

    de_available = st.session_state.get("de_results") is not None
    anova_available = st.session_state.get("multi_de_results") is not None

    if not de_available and not anova_available:
        st.info(
            "Run Differential Expression (two-group or multi-group) "
            "first — enrichment uses the significant gene list from "
            "that step."
        )
    else:
        source_options = []
        if de_available:
            source_options.append("Two-Group DE Results")
        if anova_available:
            source_options.append("Multi-Group ANOVA Results")

        gene_source = (
            st.radio("Significant Gene Source", source_options, horizontal=True)
            if len(source_options) > 1 else source_options[0]
        )

        results_table = (
            st.session_state.de_results
            if gene_source == "Two-Group DE Results"
            else st.session_state.multi_de_results
        )

        significant_genes = results_table.loc[
            results_table["Significant"], "Gene"
        ].tolist()

        st.caption(
            f"{len(significant_genes)} significant genes available "
            "from Differential Expression."
        )

        col1, col2 = st.columns(2)

        with col1:
            library_label = st.selectbox(
                "Gene Set Library",
                list(ENRICHR_LIBRARIES.keys())
            )

        with col2:
            organism = st.selectbox(
                "Organism",
                SUPPORTED_ORGANISMS
            )

        fdr_threshold = st.slider(
            "Enrichment Adjusted P-Value Threshold",
            0.01, 0.20, 0.05,
            key="enrichment_fdr"
        )

        run_clicked = st.button("Run Enrichment Analysis")

        if run_clicked:

            enrichment = EnrichmentAnalysis(significant_genes)

            with st.spinner("Querying Enrichr..."):
                results, error = enrichment.run(
                    library_label,
                    organism=organism,
                    fdr_threshold=fdr_threshold
                )

            if error:
                st.warning(error)
            else:
                st.session_state.enrichment_results = results

        if st.session_state.get("enrichment_results") is not None:

            results = st.session_state.enrichment_results

            significant_count = int(results["Significant"].sum())

            st.success(
                f"{significant_count} significant pathways found "
                f"out of {len(results)}"
            )

            display_results_table(
                    results, "enrichment",
                    long_text_columns=["Matched_Genes", "Pathway"]
                )

            plot_factory = PlotFactory(st.session_state.filtered)

            st.plotly_chart(
                plot_factory.enrichment_bar_plot(results),
                use_container_width=True
            )
with tabs[7]:

    st.header("Machine Learning")

    if st.session_state.filtered is None:
        st.info("Run Filtering first before Machine Learning.")
    else:

        samples = detect_sample_columns(st.session_state.filtered)
        gene_column = detect_gene_column(st.session_state.filtered)

        label_source = st.radio(
            "Sample Labels From",
            [
                "Two-Group DE assignment",
                "Multi-Group ANOVA assignment",
                "Manual assignment"
            ],
            horizontal=True
        )

        group_map = None

        if label_source == "Two-Group DE assignment" and st.session_state.get("de_groups"):
            group_a, group_b = st.session_state.de_groups
            group_map = {s: "Group A" for s in group_a}
            group_map.update({s: "Group B" for s in group_b})

        elif label_source == "Multi-Group ANOVA assignment" and st.session_state.get("multi_de_engine"):
            group_map = st.session_state.multi_de_engine.group_map

        else:
            st.warning(
                "No assignment found from that step yet — assign "
                "groups manually below."
            )
            group_map = group_assignment_widget(
                samples, key_prefix="ml_manual"
            )

        if group_map:

            st.write(
                "Class distribution:",
                pd.Series(list(group_map.values())).value_counts().to_dict()
            )

            feature_source = st.radio(
                "Feature Selection",
                [
                    "Use significant genes from DE/ANOVA",
                    "Auto-select top genes inside cross-validation (recommended)"
                ],
                horizontal=True
            )

            gene_list = None
            select_k = None

            if feature_source == "Use significant genes from DE/ANOVA":

                sig_genes = []

                if st.session_state.get("de_results") is not None:
                    de_r = st.session_state.de_results
                    sig_genes = de_r.loc[de_r["Significant"], "Gene"].tolist()
                elif st.session_state.get("multi_de_results") is not None:
                    mde_r = st.session_state.multi_de_results
                    sig_genes = mde_r.loc[mde_r["Significant"], "Gene"].tolist()

                if not sig_genes:
                    st.warning(
                        "No significant genes found from DE/ANOVA yet. "
                        "Falling back to automatic selection."
                    )
                    select_k = st.slider(
                        "Number of top genes to auto-select", 10, 200, 50
                    )
                else:
                    gene_list = sig_genes
                    st.caption(f"Using {len(gene_list)} significant genes as features.")

            else:
                select_k = st.slider(
                    "Number of top genes to auto-select (per fold)", 10, 200, 50
                )

            fb = FeatureBuilder(st.session_state.filtered, gene_column)

            X = None

            try:
                X, y, feature_names, nan_count = fb.build(
                    group_map, gene_list=gene_list
                )
            except ValueError as build_error:
                st.error(str(build_error))

            if X is not None:

                if nan_count > 0:
                    st.warning(
                        f"{nan_count} missing values found and filled with 0."
                    )

                ml = MLClassifier(X, y)

                min_class = int(y.value_counts().min())

                if min_class < 2:
                    st.error(
                        "At least one class has fewer than 2 samples — "
                        "cannot run cross-validation."
                    )
                else:

                    n_folds = st.slider(
                        "Cross-Validation Folds",
                        2, min_class, int(ml.suggest_folds())
                    )

                    model_options = list(ml.models.keys())

                    selected_models = st.multiselect(
                        "Models to Evaluate",
                        model_options,
                        default=model_options
                    )

                    if selected_models and st.button("Run Cross-Validation"):

                        with st.spinner("Running stratified cross-validation..."):
                            summary, detail = ml.evaluate(
                                selected_models, n_folds=n_folds, select_k=select_k
                            )

                        st.session_state.ml_summary = summary
                        st.session_state.ml_engine = ml
                        st.session_state.ml_select_k = select_k

                    if st.session_state.get("ml_summary") is not None:

                        st.success("Cross-validation complete.")
                        st.dataframe(st.session_state.ml_summary)

                        if X.shape[0] < 30:
                            st.info(
                                "With this few samples, even correctly "
                                "implemented cross-validation can show "
                                "very high or perfect scores — that "
                                "isn't automatically wrong, but treat it "
                                "as preliminary until confirmed on an "
                                "independent, larger dataset."
                            )

                        best_model = st.session_state.ml_summary.iloc[0]["Model"]

                        st.subheader(f"Confusion Matrix — {best_model}")

                        cm = st.session_state.ml_engine.confusion_matrix_cv(
                            best_model,
                            n_folds=n_folds,
                            select_k=st.session_state.ml_select_k
                        )

                        plot_factory = PlotFactory(st.session_state.filtered)

                        st.plotly_chart(
                            plot_factory.confusion_matrix_heatmap(cm),
                            use_container_width=True,
                            key="cm_classical"
                        )  
                        
                        st.subheader("ROC & Precision-Recall Curves")

                        roc_analysis = ROCAnalysis(
                            st.session_state.ml_engine,
                            best_model,
                            n_folds=n_folds,
                            select_k=st.session_state.ml_select_k
                        )
                        curves = roc_analysis.curves()

                        col_roc, col_pr = st.columns(2)

                        with col_roc:
                            st.plotly_chart(
                                plot_factory.roc_curve_plot(curves),
                                use_container_width=True,
                                key="roc_curve"
                            )

                        with col_pr:
                            st.plotly_chart(
                                plot_factory.pr_curve_plot(curves),
                                use_container_width=True,
                                key="pr_curve"
                            )

                        st.subheader("SHAP Feature Importance")

                        st.caption(
                            "Shows which genes most influenced the "
                            "model's predictions — the closer this "
                            "gets to real biological interpretation, "
                            "not just an accuracy number."
                        )

                        if st.button("Compute SHAP Importance"):

                            final_pipeline = st.session_state.ml_engine.fit_final_model(
                                best_model, select_k=st.session_state.ml_select_k
                            )

                            interpreter = ModelInterpreter(
                                final_pipeline, st.session_state.ml_engine.X
                            )

                            with st.spinner("Computing SHAP values (can take a moment)..."):
                                shap_values, shap_error = interpreter.compute()

                            if shap_error:
                                st.warning(shap_error)
                            else:
                                importance = interpreter.importance_table(shap_values)
                                st.dataframe(importance.head(20))
                                st.plotly_chart(
                                    plot_factory.shap_importance_plot(importance),
                                    use_container_width=True,
                                    key="shap_plot"
                                )
                                st.divider()
                        st.subheader("Deep Neural Network")

                        st.caption(
                            "Trains a small neural network (2 hidden "
                            "layers + dropout) with the same cross-"
                            "validation approach as above. DNNs "
                            "generally need MORE samples than classical "
                            "ML to be reliable — with under ~50 samples "
                            "per class, treat this as exploratory."
                        )

                        if st.button("Run Deep Neural Network"):

                            dnn = DNNClassifier(X, y)

                            if not dnn.is_ready():
                                st.warning(
                                    "TensorFlow is not installed. "
                                    "Run: pip install tensorflow"
                                )
                            else:
                                with st.spinner("Training neural network across folds..."):
                                    dnn_summary, dnn_cm, dnn_error = dnn.evaluate(
                                        n_folds=n_folds, epochs=100, batch_size=8
                                    )

                                if dnn_error:
                                    st.warning(dnn_error)
                                else:
                                    st.session_state.dnn_summary = dnn_summary
                                    st.session_state.dnn_cm = dnn_cm
                                    st.session_state.dnn_engine = dnn

                        if st.session_state.get("dnn_summary") is not None:

                            st.dataframe(st.session_state.dnn_summary)

                            st.plotly_chart(
                                plot_factory.confusion_matrix_heatmap(
                                    st.session_state.dnn_cm
                                ),
                                use_container_width=True,
                                key="dnn_cm"
                            )

                        st.divider()
                        st.subheader("Save Model")

                        save_name = st.text_input(
                            "Model name", value="my_first_model"
                        )

                        col_save1, col_save2 = st.columns(2)

                        with col_save1:
                            if st.button("Save Classical Model") and st.session_state.get("ml_summary") is not None:
                                final_pipeline = st.session_state.ml_engine.fit_final_model(
                                    best_model, select_k=st.session_state.ml_select_k
                                )
                                path = save_classical_model(
                                    final_pipeline, save_name,
                                    metadata={"model_type": best_model, "classes": ml.class_summary().to_dict()}
                                )
                                st.success(f"Saved to {path}")

                        with col_save2:
                            if st.button("Save DNN Model") and st.session_state.get("dnn_engine") is not None:
                                bundle, dnn_fit_error = st.session_state.dnn_engine.fit_final_model()
                                if dnn_fit_error:
                                    st.warning(dnn_fit_error)
                                else:
                                    model_path, aux_path = save_dnn_model(bundle, save_name)
                                    st.success(f"Saved to {model_path}")

                        saved = list_saved_models()
                        if saved:
                            st.caption(f"Saved models: {', '.join(saved)}")

with tabs[8]:

    st.divider()
    st.subheader("Full Publication Report (PDF)")

    st.caption(
            "Pulls together every step you've run — QC, Filtering, "
            "Normalization, DE/ANOVA, Enrichment, ML/DNN — into one "
            "PDF. Steps you haven't run yet are simply skipped."
        )

    if st.button("Generate PDF Report"):

        gene_column = detect_gene_column(st.session_state.dataset)

        figures = {}

        if st.session_state.get("enrichment_results") is not None:
            try:
                pf = PlotFactory(st.session_state.filtered)
                figures["Top Enriched Pathways"] = pf.enrichment_bar_plot(
                    st.session_state.enrichment_results
                    )
            except Exception:
                    pass

        if st.session_state.get("dnn_cm") is not None:
            try:
                pf = PlotFactory(st.session_state.filtered)
                figures["DNN Confusion Matrix"] = pf.confusion_matrix_heatmap(
                    st.session_state.dnn_cm
                    )
            except Exception:
                pass

        with st.spinner("Building PDF report..."):

            pdf_bytes = generate_pdf_report(
                dataset_summary_dict=dataset_summary(st.session_state.dataset),
                qc_dict=QualityControl(st.session_state.dataset).basic_statistics(),
                filtering_info=(
                    f"{len(st.session_state.filtered)} genes remaining after filtering."
                    if st.session_state.filtered is not None
                    else "Filtering not yet run."
                    ),
                normalization_method=(
                    "Applied" if st.session_state.normalized is not None
                        else "Not yet run"
                    ),
                de_results=st.session_state.get("de_results"),
                de_groups=st.session_state.get("de_groups"),
                anova_results=st.session_state.get("multi_de_results"),
                enrichment_results=st.session_state.get("enrichment_results"),
                ml_summary=st.session_state.get("ml_summary"),
                dnn_summary=st.session_state.get("dnn_summary"),
                figures=figures if figures else None
                )

        st.session_state.pdf_report_bytes = pdf_bytes
        st.success("Report generated.")

    if st.session_state.get("pdf_report_bytes") is not None:

            st.subheader("Preview")

            base64_pdf = base64.b64encode(
                st.session_state.pdf_report_bytes
            ).decode("utf-8")

            pdf_display = f'''
                <iframe src="data:application/pdf;base64,{base64_pdf}"
                        width="100%" height="600" type="application/pdf">
                </iframe>
            '''

            components.html(pdf_display, height=620)

            st.download_button(
                "Download PDF Report",
                st.session_state.pdf_report_bytes,
                file_name="BioAI_Report.pdf",
                mime="application/pdf"
            )

    if st.session_state.get("pdf_report_bytes") is not None:
        st.download_button(
            "Download PDF Report",
            st.session_state.pdf_report_bytes,
            file_name="BioAI_Report.pdf",
            mime="application/pdf",
            key="download_pdf_report"
        )