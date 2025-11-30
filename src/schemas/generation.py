from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    seed: Optional[int] = None


class ImageGenerateRequest(GenerationRequest):
    steps: Optional[int] = Field(30, ge=1, le=150)
    width: Optional[int] = Field(512, ge=64, le=2048)
    height: Optional[int] = Field(512, ge=64, le=2048)
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=20.0)


class VideoGenerateRequest(GenerationRequest):
    num_frames: Optional[int] = Field(24, ge=8, le=128)
    fps: Optional[int] = Field(8, ge=1, le=60)
    steps: Optional[int] = Field(50, ge=10, le=100)


class SpeechGenerateRequest(GenerationRequest):
    voice: Optional[str] = Field("default", description="Voice model")
    language: Optional[str] = Field("en", description="Language code")


class GenerationResponse(BaseModel):
    status: str
    file_path: Optional[str] = None
    generation_time: Optional[float] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DescribeRequest(BaseModel):
    file_data: bytes


class ImageDescribeResponse(BaseModel):
    description: str
    confidence: Optional[float] = None
    generation_time: Optional[float] = None


class VideoDescribeResponse(BaseModel):
    description: str
    frame_count: int
    generation_time: Optional[float] = None


class SpeechToTextResponse(BaseModel):
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
