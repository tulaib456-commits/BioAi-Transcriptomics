import streamlit as st


class Dashboard:

    @staticmethod
    def dataset_metrics(summary):

        st.subheader("Dataset Overview")

        c1, c2, c3 = st.columns(3)

        c1.metric("Genes", summary["Rows"])
        c2.metric("Samples", summary["Samples"])
        c3.metric("Missing", summary["Missing Values"])

        c1.metric("Duplicate Genes", summary["Duplicate Genes"])
        c2.metric("Columns", summary["Columns"])
        c3.metric("Gene Column", summary["Gene Column"])

    @staticmethod
    def qc_metrics(qc):

        st.subheader("Quality Control")

        c1, c2 = st.columns(2)

        c1.metric("Genes", qc["Genes"])
        c2.metric("Samples", qc["Samples"])

        c1.metric("Missing", qc["Missing"])
        c2.metric("Duplicate", qc["Duplicate"])

        c1.metric("Zero Count Genes", qc["Zero Genes"])

        c2.metric(
            "Average Library",
            round(qc["Average Library"], 2)
        )

        c1.metric(
            "Minimum Library",
            round(qc["Minimum Library"], 2)
        )

        c2.metric(
            "Maximum Library",
            round(qc["Maximum Library"], 2)
        )