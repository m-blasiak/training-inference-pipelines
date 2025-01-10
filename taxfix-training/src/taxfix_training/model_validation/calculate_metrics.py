from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model_training on test data and log metrics to MLflow.
    :param model: Trained model_training
    :param X_test: Test features
    :param y_test: Test labels
    """

    y_proba = model.predict(X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
    }
