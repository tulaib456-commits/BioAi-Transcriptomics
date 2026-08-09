import streamlit as st


class Widgets:

    @staticmethod
    def project_information(project):

        st.subheader("📁 Project Information")

        c1, c2, c3 = st.columns(3)

        c1.metric("Dataset", project["Dataset"])
        c2.metric("Genes", project["Genes"])
        c3.metric("Samples", project["Samples"])

        c1.metric("Gene Column", project["Gene Column"])
        c2.metric("File Size (KB)", project["File Size (KB)"])
        c3.metric("Status", "Ready")

    @staticmethod
    def filtering_summary(original,
                          zero_removed,
                          low_removed,
                          sparse_removed,
                          final):

        st.subheader("🧬 Filtering Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric("Original Genes", original)
        c2.metric("Zero Count Removed", zero_removed)
        c3.metric("Remaining", final)

        c1.metric("Low Count Removed", low_removed)
        c2.metric("Sparse Removed", sparse_removed)

    @staticmethod
    def normalization_information(method):

        st.subheader("Normalization")

        st.success(f"Method : {method}")