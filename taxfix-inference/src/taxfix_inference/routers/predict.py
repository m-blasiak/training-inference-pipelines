import logging
import os
from datetime import datetime, timezone
from typing import Annotated, Any

import mlflow.pyfunc
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from pydantic import BaseModel

from taxfix_inference.model.loader import get_model

router = APIRouter()

log = logging.getLogger(__name__)


class Response(BaseModel):
    prediction: float
    metadata: dict[str, Any]

    @classmethod
    def make(
        cls,
        model_version: str,
        model_instance: mlflow.pyfunc.PythonModel,
        model_prediction: float,
        user_id: str,
        user_agent: str,
    ) -> "Response":
        # Returning additional metadata for tracking/monitoring purposes
        meta = {
            "model_id": model_instance.metadata.model_uuid,
            "run_id": model_instance.metadata.run_id,
            "model_version": model_version,
            "user_id": user_id,
            "user_agent": user_agent,
            "prediction_timestamp": datetime.now(timezone.utc),
            "training_timestamp": datetime.strptime(
                model_instance.metadata.utc_time_created, "%Y-%m-%d %H:%M:%S.%f"
            ).replace(tzinfo=timezone.utc),
        }

        return cls(
            prediction=model_prediction,
            metadata=meta,
        )


class PredictionRequest(BaseModel):
    user_id: str
    features: dict[str, str | float | None]


@router.post("/predict", tags=["users"])
async def predict(
    request: PredictionRequest,
    user_agent: Annotated[str | None, Header()],
    background_tasks: BackgroundTasks,
    model: mlflow.pyfunc.PythonModel = Depends(get_model),
) -> Response:

    try:

        prediction = model.predict(pd.DataFrame([request.features]))
        background_tasks.add_task(store_inference_data, prediction)
        return Response.make(
            model_version=os.environ["MODEL_VERSION"],
            model_instance=model,
            model_prediction=prediction,
            user_id=request.user_id,
            user_agent=user_agent,
        )

    except Exception as e:
        logging.critical(f"Error while predicting: Received Exception: {e}")
        raise HTTPException(status_code=500, detail=f"Error while predicting: Received Exception: {e}")


def store_inference_data(response: Response):
    # Async call to some Pub/Sub topic that would capture the requests.
    # A PubSub <> BQ subscription will take care of storing the requests
    pass
