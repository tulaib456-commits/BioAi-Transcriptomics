import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ROCAnalysis:
    """
    One-vs-rest ROC and Precision-Recall curves using OUT-OF-FOLD
    predicted probabilities (cross_val_predict) — same honest,
    no-leakage approach as the rest of the ML module.
    """

    def __init__(self, ml_classifier, model_name, n_folds=None, select_k=None):

        self.ml = ml_classifier
        self.model_name = model_name
        self.classes = sorted(ml_classifier.y.unique())

        if n_folds is None:
            n_folds = ml_classifier.suggest_folds()

        ml_classifier.validate_folds(n_folds)

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        pipeline = ml_classifier._build_pipeline(model_name, select_k=select_k)

        self.y_proba = cross_val_predict(
            pipeline, ml_classifier.X, ml_classifier.y,
            cv=cv, method="predict_proba"
        )

    def curves(self):

        y_true = self.ml.y.values
        y_binary = label_binarize(y_true, classes=self.classes)

        if len(self.classes) == 2:
            y_binary = np.hstack([1 - y_binary, y_binary])

        results = {}

        for i, cls in enumerate(self.classes):

            fpr, tpr, _ = roc_curve(y_binary[:, i], self.y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            precision, recall, _ = precision_recall_curve(
                y_binary[:, i], self.y_proba[:, i]
            )

            results[cls] = {
                "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
                "precision": precision, "recall": recall
            }

        return results


class ModelInterpreter:
    """
    SHAP-based feature importance for a fitted pipeline.

    NOTE: could not be run against a real SHAP install during
    development (no internet in the build sandbox to install the
    'shap' package) — logic follows the documented SHAP API exactly,
    but flag immediately if this specific piece errors so it can be
    fixed fast.
    """

    def __init__(self, pipeline, X):

        self.pipeline = pipeline
        self.model = pipeline.named_steps["model"]

        transformed = X.copy()
        transformed_names = list(X.columns)

        for step_name, step in pipeline.steps[:-1]:

            transformed = step.transform(transformed)

            if step_name == "feature_select":
                mask = step.get_support()
                transformed_names = [
                    name for name, keep in zip(transformed_names, mask) if keep
                ]

        self.feature_names = transformed_names
        self.X_transformed = pd.DataFrame(
            transformed, columns=self.feature_names, index=X.index
        )

    def is_ready(self):
        return SHAP_AVAILABLE

    def compute(self, max_background=50):

        if not SHAP_AVAILABLE:
            return None, (
                "The 'shap' package is not installed. "
                "Run: pip install shap"
            )

        try:
            background = self.X_transformed

            if len(background) > max_background:
                background = background.sample(max_background, random_state=42)

            explainer = shap.Explainer(self.model.predict_proba, background)
            shap_values = explainer(self.X_transformed)

            return shap_values, None

        except Exception as error:
            return None, f"SHAP computation failed: {error}"

    def importance_table(self, shap_values):

        values = shap_values.values

        if values.ndim == 3:
            importance = np.abs(values).mean(axis=(0, 2))
        else:
            importance = np.abs(values).mean(axis=0)

        return pd.DataFrame({
            "Gene": self.feature_names,
            "Mean_Abs_SHAP": importance
        }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)