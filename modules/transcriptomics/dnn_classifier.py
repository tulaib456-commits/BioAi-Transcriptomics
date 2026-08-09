import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class DNNClassifier:
    """
    Deep neural network with stratified k-fold cross-validation —
    same honest, no-single-split evaluation used across the rest
    of the ML module. Matters even more for a DNN, since neural
    nets overfit small biological datasets very easily.
    """

    def __init__(self, X, y):
        self.X = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        self.feature_names = list(X.columns) if hasattr(X, "columns") else None

        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(y)
        self.classes = list(self.label_encoder.classes_)
        self.n_classes = len(self.classes)

    def is_ready(self):
        return TENSORFLOW_AVAILABLE

    def suggest_folds(self, max_folds=5):
        counts = pd.Series(self.y_encoded).value_counts()
        folds = min(max_folds, counts.min())
        return max(folds, 2)

    def validate_folds(self, n_folds):
        counts = pd.Series(self.y_encoded).value_counts()
        min_count = counts.min()
        if min_count < 2:
            raise ValueError(
                "At least one class has only 1 sample — cross-validation "
                "needs at least 2 samples per class."
            )
        if n_folds > min_count:
            raise ValueError(
                f"Requested {n_folds} folds but the smallest class has "
                f"only {min_count} samples. Reduce folds to at most {min_count}."
            )

    def _build_model(self, n_features, n_classes):

        model = Sequential()
        model.add(Dense(32, activation="relu", input_shape=(n_features,)))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation="relu"))
        model.add(Dropout(0.3))

        if n_classes == 2:
            model.add(Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(Dense(n_classes, activation="softmax"))
            loss = "categorical_crossentropy"

        model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        return model

    def evaluate(self, n_folds=None, epochs=100, batch_size=8, patience=10):

        if not TENSORFLOW_AVAILABLE:
            return None, None, "TensorFlow is not installed. Run: pip install tensorflow"

        if n_folds is None:
            n_folds = self.suggest_folds()

        self.validate_folds(n_folds)

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_metrics = []
        all_y_true = []
        all_y_pred = []

        for fold_index, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y_encoded)):

            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y_encoded[train_idx], self.y_encoded[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = self._build_model(X_train.shape[1], self.n_classes)

            y_train_fit = y_train if self.n_classes == 2 else to_categorical(y_train, num_classes=self.n_classes)

            early_stop = EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True)

            model.fit(X_train, y_train_fit, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])

            raw_predictions = model.predict(X_test, verbose=0)

            if self.n_classes == 2:
                y_pred = (raw_predictions.flatten() > 0.5).astype(int)
            else:
                y_pred = np.argmax(raw_predictions, axis=1)

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

            fold_metrics.append({
                "Fold": fold_index + 1,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0)
            })

        fold_df = pd.DataFrame(fold_metrics)

        summary = pd.DataFrame([{
            "Model": "Deep Neural Network",
            "Accuracy_Mean": fold_df["Accuracy"].mean(),
            "Accuracy_Std": fold_df["Accuracy"].std(),
            "F1_Mean": fold_df["F1"].mean(),
            "F1_Std": fold_df["F1"].std(),
            "Precision_Mean": fold_df["Precision"].mean(),
            "Recall_Mean": fold_df["Recall"].mean()
        }])

        labels_encoded = list(range(self.n_classes))
        cm = confusion_matrix(all_y_true, all_y_pred, labels=labels_encoded)
        cm_df = pd.DataFrame(cm, index=self.classes, columns=self.classes)

        return summary, cm_df, None

    def fit_final_model(self, epochs=100, batch_size=8, patience=10):

        if not TENSORFLOW_AVAILABLE:
            return None, "TensorFlow is not installed. Run: pip install tensorflow"

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        model = self._build_model(X_scaled.shape[1], self.n_classes)

        y_fit = self.y_encoded if self.n_classes == 2 else to_categorical(self.y_encoded, num_classes=self.n_classes)

        early_stop = EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True)

        model.fit(X_scaled, y_fit, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])

        return {"model": model, "scaler": scaler, "classes": self.classes}, None