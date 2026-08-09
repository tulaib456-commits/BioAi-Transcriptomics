import os
import joblib

MODEL_DIR = "models"


def save_classical_model(pipeline, model_name, metadata=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    joblib.dump({"pipeline": pipeline, "metadata": metadata or {}}, path)
    return path


def load_classical_model(model_name):
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def save_dnn_model(model_bundle, model_name, metadata=None):

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    aux_path = os.path.join(MODEL_DIR, f"{model_name}_aux.pkl")

    model_bundle["model"].save(model_path)

    joblib.dump({
        "scaler": model_bundle["scaler"],
        "classes": model_bundle["classes"],
        "metadata": metadata or {}
    }, aux_path)

    return model_path, aux_path


def load_dnn_model(model_name):

    from tensorflow.keras.models import load_model

    model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    aux_path = os.path.join(MODEL_DIR, f"{model_name}_aux.pkl")

    if not (os.path.exists(model_path) and os.path.exists(aux_path)):
        return None

    model = load_model(model_path)
    aux = joblib.load(aux_path)

    return {
        "model": model,
        "scaler": aux["scaler"],
        "classes": aux["classes"],
        "metadata": aux.get("metadata", {})
    }


def list_saved_models():

    if not os.path.exists(MODEL_DIR):
        return []

    names = set()

    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".pkl") and not filename.endswith("_aux.pkl"):
            names.add(filename[:-4])
        elif filename.endswith(".keras"):
            names.add(filename[:-6])

    return sorted(names)