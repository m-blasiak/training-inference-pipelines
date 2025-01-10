import logging
import os

import mlflow

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


log = logging.getLogger(__name__)


mlflow_client = None


def register_model(model_name: str, aliases: list, run_id: str):

    global mlflow_client
    if not mlflow_client:
        mlflow_client = mlflow.MlflowClient()

    source_uri = f"runs:/{run_id}/model"

    try:
        # Check if the model_training already exists
        mlflow_model = mlflow_client.get_registered_model(name=model_name)
        model_version = mlflow_client.create_model_version(name=mlflow_model.name, source=source_uri, run_id=run_id)
    except mlflow.exceptions.MlflowException as e:
        # If the model_training doesn't exist, register a new model_training and create its first version
        if e.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise e

        mlflow_model = mlflow_client.create_registered_model(name=model_name)
        model_version = mlflow_client.create_model_version(name=mlflow_model.name, source=source_uri, run_id=run_id)

    for alias in aliases:
        mlflow_client.set_registered_model_alias(name=mlflow_model.name, alias=alias, version=model_version.version)
