import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
        "Support Vector Machine": SVC(
            probability=True, class_weight="balanced", random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        )
    }


class MLClassifier:
    """
    Classical ML with stratified k-fold cross-validation.
    Feature selection (select_k) happens INSIDE each fold via the
    sklearn Pipeline, so no label information leaks from the
    validation fold into training — this is the correct, publication
    -safe way to do this, not a post-hoc train/test split.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = build_models()

    def _build_pipeline(self, model_name, select_k=None):

        steps = [("scaler", StandardScaler())]

        if select_k is not None:
            k = min(select_k, self.X.shape[1])
            steps.append(
                ("feature_select", SelectKBest(score_func=f_classif, k=k))
            )

        steps.append(("model", self.models[model_name]))

        return Pipeline(steps)

    def class_summary(self):
        return self.y.value_counts()

    def suggest_folds(self, max_folds=5):
        min_class_count = self.y.value_counts().min()
        folds = min(max_folds, min_class_count)
        return max(folds, 2)

    def validate_folds(self, n_folds):
        min_class_count = self.y.value_counts().min()
        if min_class_count < 2:
            raise ValueError(
                "At least one class has only 1 sample — "
                "cross-validation needs at least 2 samples per class."
            )
        if n_folds > min_class_count:
            raise ValueError(
                f"Requested {n_folds} folds but the smallest class "
                f"has only {min_class_count} samples. Reduce folds "
                f"to at most {min_class_count}."
            )

    def evaluate(self, model_names, n_folds=None, select_k=None):

        if n_folds is None:
            n_folds = self.suggest_folds()

        self.validate_folds(n_folds)

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        scoring = [
            "accuracy", "f1_weighted",
            "precision_weighted", "recall_weighted"
        ]

        summary_rows = []
        detail = {}

        for name in model_names:

            pipeline = self._build_pipeline(name, select_k=select_k)

            scores = cross_validate(
                pipeline, self.X, self.y,
                cv=cv, scoring=scoring, error_score="raise"
            )

            detail[name] = scores

            summary_rows.append({
                "Model": name,
                "Accuracy_Mean": scores["test_accuracy"].mean(),
                "Accuracy_Std": scores["test_accuracy"].std(),
                "F1_Mean": scores["test_f1_weighted"].mean(),
                "F1_Std": scores["test_f1_weighted"].std(),
                "Precision_Mean": scores["test_precision_weighted"].mean(),
                "Recall_Mean": scores["test_recall_weighted"].mean(),
            })

        summary = pd.DataFrame(summary_rows).sort_values(
            "F1_Mean", ascending=False
        ).reset_index(drop=True)

        return summary, detail

    def fit_final_model(self, model_name, select_k=None):

        pipeline = self._build_pipeline(model_name, select_k=select_k)

        pipeline.fit(self.X, self.y)

        return pipeline

    def confusion_matrix_cv(self, model_name, n_folds=None, select_k=None):

        if n_folds is None:
            n_folds = self.suggest_folds()

        self.validate_folds(n_folds)

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        pipeline = self._build_pipeline(model_name, select_k=select_k)

        y_pred = cross_val_predict(pipeline, self.X, self.y, cv=cv)

        labels = sorted(self.y.unique())

        cm = confusion_matrix(self.y, y_pred, labels=labels)

        return pd.DataFrame(cm, index=labels, columns=labels)