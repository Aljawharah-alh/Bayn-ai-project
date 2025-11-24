from fastapi import FastAPI
from app.api.v1 import router

app = FastAPI(
    title="Bayn API",
    version="1.0.0"
)

app.include_router(router.api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Landmark Audio API is running 🚀"}
