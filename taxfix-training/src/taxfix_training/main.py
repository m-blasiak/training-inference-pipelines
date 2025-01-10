import mlflow
from sklearn.model_selection import train_test_split

from taxfix_training.data_access_layer.csv_loader import load_dataset
from taxfix_training.model_training.train import train
from taxfix_training.model_validation.calculate_metrics import evaluate_model
from taxfix_training.utils import ROOT_DIR, register_model


def main(data_path: str, cat_features: list[str], target: str):
    """
    Entrypoint to the training script.
    Normally this would be orchestrated by some regular re-training job for example by an Airflow DAG
    The steps involved are:
    - splitting the data into train, test & validation script.
    - training the model using the dedicated training script
    - evaluating the model by triggering the dedicated evaluation script
    - registering the model as a "champion" in MLFlow. Normally, I'd add some champion/challenger logic

    :param data_path:  path to the training dataset
    :param cat_features: List of categorical features expected by the model
    :param target: Name of the column containing the targets in the dataset
    :return:
    """

    df = load_dataset(data_path)
    X = df.drop(columns=[target])
    y = df[target]

    # Split into train + validation, and test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split train + validation into separate training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    df.to_json()
    model = train(X_train=X_train, y_train=y_train, cat_features=cat_features, hyperparameters={})

    X_val.head().to_json("/validation-data/x_val.json", orient="index", indent=2)
    y_val.to_json("/validation-data/y_val.json", orient="index", indent=1)

    model_metrics = evaluate_model(model, X_test, y_test)
    for metric_name, metric_value in model_metrics.items():
        mlflow.log_metric(metric_name, metric_value, run_id=model.metadata.run_id)

    # A champion/challenger logic could be introduced here. Where newly trained models would be tagged as
    # challengers. Then, a separate script would compare the newly trained champion to a challenger and potentially
    # create a new champion
    register_model(model_name="taxfix_classifier", aliases=["champion"], run_id=model.metadata.run_id)


if __name__ == "__main__":
    TARGET = "completed_filing"
    DATA_PATH = f"{ROOT_DIR}/logdir/dataset.csv"
    CAT_FEATURES = ["employment_type", "marital_status", "device_type", "referral_source"]

    main(data_path=DATA_PATH, cat_features=CAT_FEATURES, target=TARGET)
