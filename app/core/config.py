from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # app/core -> app

class Settings(BaseSettings):
    VISIONSTORY_API_KEY: str
    VISIONSTORY_AVATAR_ID: str
    NGROK_INFERENCE_URL: str

    DESCRIPTIONS_FILE_PATH: Path = BASE_DIR / "data" / "landmarks_descriptions.csv"

    class Config:
        env_file = ".env"

settings = Settings()