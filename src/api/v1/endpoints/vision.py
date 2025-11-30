from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ....schemas.generation import ImageDescribeResponse, VideoDescribeResponse
from ....services.vision_service import get_vision_service
from ...dependencies import verify_api_key

router = APIRouter(prefix="/vision", tags=["Vision"])


@router.post("/describe", response_model=ImageDescribeResponse)
async def describe_image(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    api_key: str = Depends(verify_api_key),
    service=Depends(get_vision_service),
):
    """
    Get a caption/description for an image.
    Optionally provide a prompt to guide the description.
    """
    image_data = await file.read()
    result = await service.describe_image(image_data, prompt)
    return ImageDescribeResponse(**result)


@router.post("/describe_video", response_model=VideoDescribeResponse)
async def describe_video(
    file: UploadFile = File(...),
    num_frames: int = Form(8),
    api_key: str = Depends(verify_api_key),
    service=Depends(get_vision_service),
):
    """
    Get a description for a video by sampling and analyzing frames.

    Args:
        file: Video file (mp4, avi, mov, webm, mkv)
        num_frames: Number of frames to sample for analysis (default: 8)
    """
    video_data = await file.read()
    result = await service.describe_video(video_data, num_frames)
    return VideoDescribeResponse(**result)


@router.post("/vqa")
async def visual_question_answering(
    file: UploadFile = File(...),
    question: str = Form(...),
    api_key: str = Depends(verify_api_key),
    service=Depends(get_vision_service),
):
    """
    Visual Question Answering - ask a question about an image.

    Args:
        file: Image file
        question: Question to ask about the image
    """
    image_data = await file.read()
    result = await service.answer_question(image_data, question)
    return ImageDescribeResponse(**result)
