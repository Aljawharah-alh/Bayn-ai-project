import requests
from fastapi import HTTPException
from app.core.config import settings

class VisionStoryService:
    """
    Service class for interacting with the VisionStory AI API.
    """
    def __init__(self):
        # VisionStory uses a custom header for the plain text API key
        self.api_key = settings.VISIONSTORY_API_KEY
        self.base_url = "https://openapi.visionstory.ai/api/v1/video"
        self.headers = {
            # <<< CORRECT AUTHENTICATION FOR VISIONSTORY >>>
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        # Avatar ID needs to be configured in settings (using your example's ID as placeholder)
        self.avatar_id = settings.VISIONSTORY_AVATAR_ID 
        
        if not self.api_key or not self.avatar_id:
             raise RuntimeError("VisionStory service configuration is missing (API Key or Avatar ID).")

    def create_video_job(self, text_script: str) -> str:
        """
        Starts an asynchronous video generation job and returns the video_id.
        """
        if not text_script or not text_script.strip():
            raise HTTPException(status_code=400, detail="Empty text provided for video script.")

        # Payload structure
        payload = {
            "model_id": "vs_talk_v1",
            "avatar_id": self.avatar_id,
            "text_script": {
                "text": text_script,
                # Example voice_id from your script; change to an Arabic voice if available/desired
                "voice_id": "Echo" 
            },
            "aspect_ratio": "1:1",
            "resolution": "480p"
        }

        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status() # Raises HTTPError for 4xx/5xx responses

            resp_data = response.json()
            # VisionStory returns the job ID as 'video_id' inside the 'data' field
            video_id = resp_data.get("data", {}).get("video_id")
            
            if not video_id:
                raise HTTPException(
                    status_code=500,
                    detail=f"VisionStory API returned success but no video_id. Response: {resp_data}"
                )

            return video_id

        except requests.exceptions.RequestException as e:
            print(f">>> VisionStory API Request Error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start VisionStory video job: {e}"
            )
        
    def get_video_status(self, video_id: str) -> dict:
        """
        Checks the status of the VisionStory video generation job.
        """
        if not video_id:
            raise HTTPException(status_code=400, detail="Video ID is required.")

        params = {"video_id": video_id}
        
        try:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            resp_data = response.json()
            # VisionStory returns the status data inside the 'data' field
            return resp_data.get("data", {})
        
        except requests.exceptions.RequestException as e:
            print(f">>> VisionStory Status Request Error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve VisionStory job status: {e}"
            )