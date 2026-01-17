import pandas as pd
from functools import lru_cache
from app.core.config import settings

@lru_cache(maxsize=1)
def _load_landmark_descriptions() -> dict[str, str]:
    print("Loading descriptions from:", settings.DESCRIPTIONS_FILE_PATH)

    df = pd.read_csv(settings.DESCRIPTIONS_FILE_PATH)

    descriptions = {
        str(row["name"]).strip().lower(): str(row["description"]).strip()
        for _, row in df.iterrows()
    }
    return descriptions

def get_landmark_description(landmark_name: str) -> str:
    landmark_descriptions = _load_landmark_descriptions()
    key = str(landmark_name).strip().lower()
    return landmark_descriptions.get(
        key,
        "No description available for this landmark.",
    )
