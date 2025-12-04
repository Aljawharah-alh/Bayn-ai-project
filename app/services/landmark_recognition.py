import asyncio
from fastapi import UploadFile
import requests
from app.core.config import settings


INFERENCE_API_URL = settings.NGROK_INFERENCE_URL 

class LandmarkRecognitionService:
    """
    This service now acts as a client that sends the image to the 
    remote Colab endpoint for inference.
    """

    def __init__(self):
        # We no longer load the model locally.
        print(f"Connecting to remote inference service at: {INFERENCE_API_URL}")
        if not INFERENCE_API_URL:
            raise RuntimeError("NGROK_INFERENCE_URL is missing in settings. Please configure.")

    async def recognize(self, image: UploadFile) -> str:
        """
        Sends the uploaded image to the remote inference API and returns the landmark name.
        """
        # Read the image content asynchronously
        image_bytes = await image.read()
        
        # Prepare the file payload for the POST request
        files = {'image': (image.filename, image_bytes, image.content_type)}

        # Run the synchronous HTTP request in a separate thread
        def _call_remote_model():
            try:
                # The endpoint expects a POST request with the image under the 'image' file field
                response = requests.post(INFERENCE_API_URL, files=files, timeout=30)
                response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
                
                data = response.json()
                
                # Check for prediction success
                if 'landmark_name' in data:
                    return data['landmark_name']
                else:
                    raise Exception(f"Prediction failed: {data.get('error', 'Unknown response from remote API')}")

            except requests.exceptions.RequestException as e:
                print(f"Remote API call failed: {e}")
                raise

        # Use asyncio.to_thread to run the blocking requests.post call
        try:
            landmark_name = await asyncio.to_thread(_call_remote_model)
            return landmark_name
        except Exception as e:
            # Fallback or error handling
            print(f"Error during remote recognition: {e}")
            # You might want to return a specific error landmark or raise an HTTPException
            return "recognition_failed"