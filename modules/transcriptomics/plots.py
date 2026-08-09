import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from modules.transcriptomics.validator import detect_sample_columns


class PlotFactory:

    def __init__(self, dataframe):

        self.df = dataframe

        self.samples = detect_sample_columns(dataframe)

    def library_size_plot(self):

        library = self.df[self.samples].sum()

        return px.bar(
            x=library.index,
            y=library.values,
            labels={
                "x": "Sample",
                "y": "Library Size"
            },
            title="Library Size"
        )

    def histogram(self):

        values = self.df[self.samples].values.flatten()

        return px.histogram(
            x=values,
            nbins=100,
            title="Read Count Distribution"
        )

    def boxplot(self):

        melted = self.df[self.samples].melt()

        return px.box(
            melted,
            x="variable",
            y="value",
            title="Sample Distribution"
        )

    def density_plot(self):

        values = self.df[self.samples].values.flatten()

        return px.histogram(
            x=values,
            histnorm="probability density",
            marginal="violin",
            title="Density Distribution"
        )

    def correlation_heatmap(self):

        corr = self.df[self.samples].corr()

        return px.imshow(
            corr,
            text_auto=".2f",
            title="Sample Correlation"
        )

    def pca_plot(self, group_map=None, top_n_genes=2000):
        """
        Real PCA on samples using the top N most variable genes.

        group_map : dict {sample_name: group_label}, optional.
                    Used to color points by group (e.g. mock vs CoV2).
        """

        data = self.df[self.samples].astype(float)

        variances = data.var(axis=1)

        top_genes = variances.sort_values(
            ascending=False
        ).head(top_n_genes).index

        matrix = data.loc[top_genes].T  # samples as rows

        scaled = StandardScaler().fit_transform(matrix)

        n_components = min(2, scaled.shape[0], scaled.shape[1])

        pca = PCA(n_components=n_components)

        scores = pca.fit_transform(scaled)

        result = pd.DataFrame(
            scores,
            columns=[f"PC{i + 1}" for i in range(n_components)],
            index=self.samples
        )

        result["Sample"] = self.samples

        if group_map:
            result["Group"] = [
                group_map.get(sample, "Unassigned")
                for sample in self.samples
            ]
            color = "Group"
        else:
            color = None

        explained = pca.explained_variance_ratio_ * 100

        x_label = f"PC1 ({explained[0]:.1f}% variance)"

        y_label = (
            f"PC2 ({explained[1]:.1f}% variance)"
            if n_components > 1
            else ""
        )

        fig = px.scatter(
            result,
            x="PC1",
            y="PC2" if n_components > 1 else None,
            color=color,
            text="Sample",
            title="PCA of Samples",
            labels={"PC1": x_label, "PC2": y_label}
        )

        fig.update_traces(
            textposition="top center",
            marker=dict(size=12)
        )

        return fig

    def volcano_plot(self, de_results, log2fc_threshold=1.0):
        """
        Volcano plot from a DifferentialExpression.run() result table.
        """

        data = de_results.copy()

        data["-log10(Adj P)"] = -np.log10(
            data["Adj_P_Value"].replace(0, 1e-300)
        )

        data["Direction"] = "Not Significant"

        data.loc[
            (data["Significant"]) & (data["Log2FC"] > 0),
            "Direction"
        ] = "Up in Group B"

        data.loc[
            (data["Significant"]) & (data["Log2FC"] < 0),
            "Direction"
        ] = "Down in Group B"

        fig = px.scatter(
            data,
            x="Log2FC",
            y="-log10(Adj P)",
            color="Direction",
            hover_name="Gene",
            title="Volcano Plot",
            color_discrete_map={
                "Not Significant": "lightgray",
                "Up in Group B": "crimson",
                "Down in Group B": "royalblue"
            }
        )

        fig.add_vline(x=log2fc_threshold, line_dash="dash")
        fig.add_vline(x=-log2fc_threshold, line_dash="dash")

        return fig
        
    def enrichment_bar_plot(self, results, top_n=15):

        data = results.head(top_n).copy()

        data["-log10(Adj P)"] = -np.log10(
            data["Adj_P_Value"].replace(0, 1e-300)
        )

        data = data.sort_values("-log10(Adj P)")

        fig = px.bar(
            data,
            x="-log10(Adj P)",
            y="Pathway",
            orientation="h",
            title="Top Enriched Pathways",
            color="-log10(Adj P)",
            color_continuous_scale="Viridis"
        )

        fig.update_layout(showlegend=False)

        return fig
        
    def gene_boxplot(self, expression_table, gene_name):

        fig = px.box(
            expression_table,
            x="Group",
            y="Expression",
            points="all",
            color="Group",
            title=f"{gene_name} Expression by Group"
        )

        fig.update_layout(showlegend=False)

        return fig
    def confusion_matrix_heatmap(self, cm_df):

        fig = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix (out-of-fold predictions)",
            labels=dict(x="Predicted", y="Actual", color="Count")
        )

        return fig

    def roc_curve_plot(self, curves_dict):

        fig = go.Figure()

        for cls, data in curves_dict.items():
            fig.add_trace(go.Scatter(
                x=data["fpr"], y=data["tpr"], mode="lines",
                name=f"{cls} (AUC={data['roc_auc']:.3f})"
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Random chance"
        ))

        fig.update_layout(
            title="ROC Curve (one-vs-rest)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )

        return fig

    def pr_curve_plot(self, curves_dict):

        fig = go.Figure()

        for cls, data in curves_dict.items():
            fig.add_trace(go.Scatter(
                x=data["recall"], y=data["precision"], mode="lines",
                name=str(cls)
            ))

        fig.update_layout(
            title="Precision-Recall Curve (one-vs-rest)",
            xaxis_title="Recall",
            yaxis_title="Precision"
        )

        return fig

    def shap_importance_plot(self, importance_table, top_n=20):

        data = importance_table.head(top_n).sort_values("Mean_Abs_SHAP")

        fig = px.bar(
            data, x="Mean_Abs_SHAP", y="Gene", orientation="h",
            title="Top Genes by SHAP Importance",
            color="Mean_Abs_SHAP", color_continuous_scale="Viridis"
        )

        fig.update_layout(showlegend=False)

        return fig

    def top_genes_heatmap(self, expression_matrix):

        z = expression_matrix.sub(
            expression_matrix.mean(axis=1), axis=0
        ).div(
            expression_matrix.std(axis=1).replace(0, 1), axis=0
        )

        fig = px.imshow(
            z,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            labels=dict(x="Sample", y="Gene", color="Z-score"),
            title="Top Significant Genes (z-score normalized)"
        )

        return fig

    def library_size_bar(self, library_sizes_dict):

        data = pd.DataFrame({
            "Sample": list(library_sizes_dict.keys()),
            "Library Size": list(library_sizes_dict.values())
        })

        median_size = data["Library Size"].median()

        fig = px.bar(
            data, x="Sample", y="Library Size",
            title="Library Size per Sample"
        )
        fig.add_hline(
            y=median_size, line_dash="dash",
            annotation_text="Median"
        )
        fig.update_xaxes(tickangle=45)

        return fig

    def correlation_heatmap(self, correlation_matrix):

        fig = px.imshow(
            correlation_matrix,
            color_continuous_scale="RdBu_r",
            zmin=correlation_matrix.values.min(),
            zmax=1.0,
            title="Sample-Sample Correlation (Spearman)",
            labels=dict(color="Correlation")
        )

        return fig

    def rle_plot(self, log_cpm_df):

        median_per_gene = log_cpm_df.median(axis=1)
        rle = log_cpm_df.sub(median_per_gene, axis=0)

        fig = go.Figure()

        for sample in rle.columns:
            fig.add_trace(go.Box(y=rle[sample], name=sample))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            title="Relative Log Expression (RLE) — should center near 0",
            yaxis_title="RLE",
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)

        return fig

    def expression_density(self, log_cpm_df, max_genes=2000):

        if len(log_cpm_df) > max_genes:
            log_cpm_df = log_cpm_df.sample(max_genes, random_state=42)

        fig = go.Figure()

        for sample in log_cpm_df.columns:
            fig.add_trace(go.Violin(
                y=log_cpm_df[sample], name=sample, box_visible=False,
                meanline_visible=True, points=False
            ))

        fig.update_layout(
            title="Expression Distribution per Sample (log2 CPM)",
            yaxis_title="log2(CPM + 1)",
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)

        return fig
