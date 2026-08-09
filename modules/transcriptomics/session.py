import streamlit as st


class SessionManager:

    @staticmethod

    def initialize():

        defaults = {

            "dataset": None,

            "metadata": None,

            "filtered": None,

            "normalized": None,

            "tracker": None,

            "dataset_path": None,

            "de_results": None,

            "multi_de_results": None,

            "multi_de_engine": None,

            "enrichment_results": None,

            "de_groups": None,

            "ml_summary": None,

            "ml_engine": None,

            "ml_select_k": None,

            "dnn_summary": None,

            "dnn_cm": None,

            "dnn_engine": None,

            "pdf_report_bytes": None,
        
            "qc_corr_matrix": None,

            "qc_density_fig": None,

            "norm_rle_fig": None,

            "ml_pipeline": None,

        }

        for key, value in defaults.items():

            if key not in st.session_state:

                st.session_state[key] = value

    @staticmethod

    def reset():

        for key in list(

            st.session_state.keys()

        ):

            del st.session_state[key]