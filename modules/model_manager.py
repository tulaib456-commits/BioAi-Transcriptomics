import joblib
import os


def save_model(model, model_name):

    os.makedirs("models", exist_ok=True)

    filename = f"models/{model_name}.pkl"

    joblib.dump(model, filename)

    return filename


def load_model(model_name):

    filename = f"models/{model_name}.pkl"

    if os.path.exists(filename):

        return joblib.load(filename)

    return None