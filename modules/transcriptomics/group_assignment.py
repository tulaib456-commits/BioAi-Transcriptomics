import pandas as pd
import streamlit as st


def group_assignment_widget(samples, key_prefix, suggested_map=None):
    """
    Group-assignment UI with three ways to label samples:
      1. Bulk keyword assign — type a keyword + group label, apply
         to every matching sample name at once (handles 285 samples
         in ~10 clicks instead of 285 manual edits)
      2. Upload a Sample,Group mapping CSV (if you already track
         this in a spreadsheet)
      3. Manual per-row editing (for final touch-ups / exceptions)

    Returns {sample: group_label}, non-empty assignments only.
    """

    state_key = f"{key_prefix}_group_table"

    if state_key not in st.session_state:
        st.session_state[state_key] = pd.DataFrame({
            "Sample": samples,
            "Group": [
                suggested_map.get(s, "") if suggested_map else ""
                for s in samples
            ]
        })

    st.caption(
        "Assign a group to each sample. For large datasets, use "
        "bulk keyword assignment below instead of editing row by row."
    )

    with st.expander("Bulk assign by keyword", expanded=True):

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            keyword = st.text_input(
                "Sample name contains", key=f"{key_prefix}_keyword"
            )

        with col2:
            label = st.text_input(
                "Assign group label", key=f"{key_prefix}_label"
            )

        with col3:
            st.write("")
            st.write("")
            apply_clicked = st.button("Apply", key=f"{key_prefix}_apply")

        if apply_clicked and keyword and label:

            table = st.session_state[state_key]
            mask = table["Sample"].str.contains(keyword, case=False, regex=False)
            matched = int(mask.sum())

            table.loc[mask, "Group"] = label
            st.session_state[state_key] = table

            if matched > 0:
                st.success(f"Assigned '{label}' to {matched} matching sample(s).")
            else:
                st.warning(f"No sample names contain '{keyword}'.")

    with st.expander("Upload a Sample-Group mapping CSV", expanded=False):

        st.caption(
            "CSV with two columns: 'Sample' and 'Group'. Sample "
            "values must match your dataset's column names exactly."
        )

        mapping_file = st.file_uploader(
            "Upload mapping CSV", type=["csv"], key=f"{key_prefix}_csv"
        )

        if mapping_file is not None:

            try:
                mapping_df = pd.read_csv(mapping_file)

                if "Sample" not in mapping_df.columns or "Group" not in mapping_df.columns:
                    st.error("CSV must have 'Sample' and 'Group' columns.")
                else:
                    table = st.session_state[state_key]
                    mapping_lookup = dict(zip(mapping_df["Sample"], mapping_df["Group"]))

                    matched = 0
                    for i, row in table.iterrows():
                        if row["Sample"] in mapping_lookup:
                            table.loc[i, "Group"] = mapping_lookup[row["Sample"]]
                            matched += 1

                    st.session_state[state_key] = table
                    st.success(f"Matched and filled {matched} of {len(table)} samples.")

            except Exception as error:
                st.error(f"Could not read CSV: {error}")

    edited_table = st.data_editor(
        st.session_state[state_key],
        column_config={"Sample": st.column_config.TextColumn(disabled=True)},
        hide_index=True,
        use_container_width=True,
        key=f"{key_prefix}_editor"
    )

    st.session_state[state_key] = edited_table

    return {
        row["Sample"]: row["Group"]
        for _, row in edited_table.iterrows()
        if row["Group"]
    }