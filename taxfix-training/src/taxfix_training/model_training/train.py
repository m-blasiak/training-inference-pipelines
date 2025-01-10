import logging
import pickle
from datetime import datetime
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    cat_features: list[str],
    hyperparameters: dict[str, Any],
) -> mlflow.models.Model:
    """

    :param X_train: Training dataset
    :param y_train: Targets for training dataset
    :param cat_features: list of categorical features
    :param hyperparameters: hyperparams to be passed to the model
    :return:
    """

    mlflow.set_experiment("taxfix_case_study")
    mlflow.start_run(run_name=f"taxfix_case_study_training_{datetime.now()}")
    num_features = [e for e in list(X_train.columns) if e not in cat_features]

    model = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[("num", MinMaxScaler(), num_features), ("cat", OneHotEncoder(), cat_features)]
                ),
            ),
            ("classifier", RandomForestClassifier(**hyperparameters)),
        ]
    )

    model.fit(X_train, y_train)

    preprocessor = model.named_steps["preprocessor"]
    feature_importance_df = pd.DataFrame(
        {
            "Feature": num_features + list(preprocessor.transformers_[1][1].get_feature_names_out(cat_features)),
            "Importance": model.named_steps["classifier"].feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    mlflow.log_table(feature_importance_df, "model/feature_importance.json")
    with open("taxfix_training/logdir/model.pkl", "wb") as f:
        pickle.dump(model, f)

    artifacts = {
        "model_path": "taxfix_training/logdir/model.pkl",
    }

    class ModelWrapper(mlflow.pyfunc.PythonModel):
        """
        A custom wrapper for MLFlow's PyFunc model to integrate feature transformations
        (e.g., one-hot encoding) and prediction logic in a consistent pipeline.
        It also helps with keeping a consistent interface when changing model algorithms
        """

        def load_context(self, context):
            """
            Method responsible for loading artifacts required for the model to work
            :param context:
            :return:
            """
            with open(context.artifacts["model_path"], "rb") as f:
                self.model = pickle.load(f)

        def predict(self, context: Any, model_input, params=None) -> np.array:
            """
            This is the method that will be called during inference
            :param context:
            :param model_input: Data to make predictions on
            :param params:
            :return: Numpy array with prediction probabilities
            """

            predictions = self.model.predict_proba(model_input)[:, 1]
            return np.array(predictions)

    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ModelWrapper(),
        artifacts=artifacts,
    )
    mlflow.end_run()
    logging.getLogger(__name__)
    return mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
