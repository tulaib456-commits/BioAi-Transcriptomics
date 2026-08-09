import streamlit as st


def display_results_table(dataframe, key_prefix, long_text_columns=None):
    """
    Reusable results table: a "how many rows" selector, plus
    explicit column width control so long text columns (e.g. a
    semicolon-joined gene list) don't crowd adjacent columns.
    """

    display_count = st.selectbox(
        "Show results",
        [10, 30, 50, 100, "All"],
        index=2,
        key=f"{key_prefix}_display_count"
    )

    display_df = (
        dataframe if display_count == "All"
        else dataframe.head(display_count)
    )

    column_config = {}

    if long_text_columns:
        for col in long_text_columns:
            if col in display_df.columns:
                column_config[col] = st.column_config.TextColumn(
                    col, width="large"
                )

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config=column_config if column_config else None,
        key=f"{key_prefix}_table"
    )

    st.caption(f"Showing {len(display_df)} of {len(dataframe)} total rows.")