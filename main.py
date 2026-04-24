from fastapi import FastAPI

from routes import health, upload, train, predict

app = FastAPI(title="Iris MLOps API")

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(train.router)
app.include_router(predict.router)