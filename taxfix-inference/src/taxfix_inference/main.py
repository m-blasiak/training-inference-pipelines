import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from taxfix_inference.model.loader import initialize_model
from taxfix_inference.routers import predict

log = logging.getLogger(__name__)
log.info("Application is starting up...")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_model()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(predict.router)


@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok"})
