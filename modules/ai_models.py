from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return {
        
        "model_object": model,
        
        "accuracy": accuracy_score(y_test, predictions),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

    }


def train_all_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Decision Tree":
            DecisionTreeClassifier(random_state=42),

        "Random Forest":
            RandomForestClassifier(random_state=42),

        "Support Vector Machine":
            SVC(),

        "K Nearest Neighbor":
            KNeighborsClassifier(n_neighbors=3)

    }

    results = {}

    for name, model in models.items():

        results[name] = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    return results