import logging
import os
import time

import mlflow

model = None
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def initialize_model():
    global model

    start_time = time.time()
    tracking_server = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    model_alias = os.environ.get("MODEL_VERSION", "champion")

    if tracking_server == "http://127.0.0.1:5000":
        log.info("Using Local MLFlow tracking server!")

    os.environ["MLFLOW_TRACKING_URI"] = tracking_server
    os.environ["MODEL_VERSION"] = model_alias

    log.info("Loading MLFlow model")
    try:
        log.info(f"Loading {model_alias} model")
        model = mlflow.pyfunc.load_model(f"models:/taxfix_classifier@{model_alias}")
    except Exception as e:
        log.error(f"Error loading {model_alias} model: {e}")
        exit(1)

    log.info("Model loaded in : %.2f seconds", time.time() - start_time)


def get_model() -> mlflow.pyfunc.PythonModel:
    return model
