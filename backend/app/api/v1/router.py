from fastapi import APIRouter

from .endpoints import health
from .endpoints import landmarks_from_image

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(
    landmarks_from_image.router,
    prefix="/landmarks",
    tags=["Landmarks"],
)