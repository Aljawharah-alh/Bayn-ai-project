# File: app\api\v1\endpoints\landmarks_from_image.py
from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from app.services.landmark_recognition import LandmarkRecognitionService
from app.services.landmark_description_service import get_landmark_description
from app.services.visionstory_service import VisionStoryService  

router = APIRouter()

# 1. Endpoint to START the video generation job (asynchronous)
@router.post("/start-video-job")
async def start_video_job(
    image: UploadFile = File(...),
    recognizer: LandmarkRecognitionService = Depends(LandmarkRecognitionService),
    vs_service: VisionStoryService = Depends(VisionStoryService), 
):
    """
    Recognizes the landmark and starts the VisionStory video generation job.
    Returns the video_id for the client to poll.
    """
    landmark_name = await recognizer.recognize(image)
    description = get_landmark_description(landmark_name)
    
    print(">>> Landmark recognized:", landmark_name)
    print(">>> Description to be used:", description[:50] + "...")

    # 2. Start VisionStory Job
    try:
        # Renamed variable to reflect VisionStory's 'video_id'
        video_id = vs_service.create_video_job(description) 
        
        print(">>> VisionStory Job Started. ID:", video_id)
        
        return JSONResponse(
            status_code=202, # 202 Accepted
            content={"video_id": video_id, "landmark_name": landmark_name} # <<< RENAMED KEY
        )
    except Exception as e:
        print(f"Error starting VisionStory job: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Failed to start video generation service."}
        )

# 2. Endpoint to CHECK the video generation status
@router.get("/video-status/{video_id}") 
def get_video_status(video_id: str, vs_service: VisionStoryService = Depends(VisionStoryService)):
    """
    Checks the status of the VisionStory job using the provided video_id.
    """
    status_data = vs_service.get_video_status(video_id)
    
    # VisionStory status can be: waiting, processing, completed, failed
    status = status_data.get("status")
    video_url = status_data.get("video_url")
    
    if status == "created" and video_url: # <<< USED STANDARD 'completed' STATUS
        print(f">>> Job {video_id} COMPLETED. URL: {video_url}")
        # Note: Your example uses "created", but 'completed' is standard final status. 
        return {"status": "done", "video_url": video_url}
    
    elif status == "failed":
        print(f">>> Job {video_id} FAILED.")
        return {"status": "failed", "error": status_data.get("error_msg")} # VisionStory may use error_msg
        
    print(f">>> Job Outside if {video_id} current status:", status)
    return {"status": status}