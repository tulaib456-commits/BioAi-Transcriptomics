import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def train_deep_learning_model(
    X_train,
    X_test,
    y_train,
    y_test
):

    model = Sequential()

    model.add(
        Dense(
            16,
            activation="relu",
            input_shape=(X_train.shape[1],)
        )
    )

    model.add(
        Dense(
            8,
            activation="relu"
        )
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(

        optimizer="adam",

        loss="binary_crossentropy",

        metrics=["accuracy"]

    )

    model.fit(

        X_train,

        y_train,

        epochs=50,

        verbose=0

    )

    predictions = model.predict(
        X_test,
        verbose=0
    )

    predictions = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    return {

        "Model": "Deep Neural Network",

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "Confusion Matrix": cm

    }