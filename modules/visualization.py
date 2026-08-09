import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def show_accuracy_chart(comparison_df):

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        comparison_df["Model"],
        comparison_df["Accuracy"]
    )

    ax.set_title("Model Accuracy Comparison")

    ax.set_ylabel("Accuracy")

    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    st.pyplot(fig)