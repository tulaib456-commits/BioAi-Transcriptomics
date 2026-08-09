from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(X, y):
    """
    Complete preprocessing pipeline

    Parameters
    ----------
    X : pandas DataFrame
    y : pandas Series

    Returns
    -------
    Dictionary containing all processed datasets
    """

    # -------------------------
    # Remove ID Columns
    # -------------------------

    possible_id_columns = [
        "Patient_ID",
        "Sample_ID",
        "ID",
        "Subject_ID"
    ]

    for col in possible_id_columns:

        if col in X.columns:

            X = X.drop(col, axis=1)

    # -------------------------
    # Keep Numeric Columns
    # -------------------------

    X_numeric = X.select_dtypes(
        include=["int64", "float64"]
    )

    # -------------------------
    # Normalize
    # -------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_numeric
    )

    # -------------------------
    # Train Test Split
    # -------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X_scaled,

        y,

        test_size=0.2,

        random_state=42

    )

    return {

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "feature_names": list(
            X_numeric.columns
        ),

        "feature_count": X_numeric.shape[1],

        "training_samples": len(
            X_train
        ),

        "testing_samples": len(
            X_test
        )

    }