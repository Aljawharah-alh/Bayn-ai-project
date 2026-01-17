# File: app\main.py (MODIFIED)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # <<< NEW IMPORT
from app.api.v1 import router

app = FastAPI(
    title="Bayn API",
    version="1.0.0"
)

# === CORS CONFIGURATION FIX ===
origins = [
    # 💡 Your FastAPI is at 8000. Your browser is fetching from 5500 (VS Code Live Server).
    # We must allow the front-end origin (5500) to talk to the backend.
    "http://localhost:5500", 
    "http://127.0.0.1:5500",
    # If you need to access the API directly, you can include its own origin too
    # "http://127.0.0.1:8000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows POST, GET, etc.
    allow_headers=["*"], # Allows all headers
)
# === END CORS CONFIGURATION ===

app.include_router(router.api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Landmark Audio API is running"}