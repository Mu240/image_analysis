import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import base64
import logging
from typing import List, Dict, Any
from pydantic import BaseModel
from image_analysis import OpenAIImageAnalyzer, ImageInput, OpenAIParams, ImageManipulationParams
import os
import httpx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="OpenAI Image Analysis API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants aligned with image_analysis.py
AVAILABLE_MODELS = ["gpt-4.1", "gpt-4o", "gpt-4-turbo"]
DEFAULT_MODEL = "gpt-4.1"  # Aligned with image_analysis.py
VISION_DETAIL_LEVEL = "low"  # Aligned with image_analysis.py

# Pydantic model for request validation
class AnalysisRequest(BaseModel):
    user_prompt: str
    system_prompt: str
    max_tokens: int = 600  # Aligned with image_analysis.py
    detail_level: str = VISION_DETAIL_LEVEL  # Default to low
    event_level: bool = True
    api_key: str
    model: str = DEFAULT_MODEL

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "OpenAI Image Analysis API is running"}

@app.post("/analyze")
async def analyze_images(
    images: List[UploadFile] = File(...),
    user_prompt: str = Form(...),
    system_prompt: str = Form(...),
    max_tokens: int = Form(600),  # Aligned with image_analysis.py
    detail_level: str = Form(VISION_DETAIL_LEVEL),
    event_level: bool = Form(True),
    api_key: str = Form(...),
    model: str = Form(DEFAULT_MODEL)
):
    """Analyze uploaded images using OpenAI GPT-4 Vision with system prompt-defined schema"""
    try:
        start_time = time.time()
        logger.info(f"Received analysis request: {len(images)} images, event_level={event_level}, model={model}")

        # Validate inputs
        if not images:
            raise HTTPException(status_code=400, detail="No images provided")
        if len(images) > 10:
            raise HTTPException(status_code=400, detail="Maximum of 10 images allowed")
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        if model not in AVAILABLE_MODELS:
            logger.warning(f"Invalid model {model}, using default {DEFAULT_MODEL}")
            model = DEFAULT_MODEL

        # Convert uploaded files to ImageInput objects
        image_inputs = []
        for idx, image in enumerate(images):
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"File {image.filename} is not an image")
            content = await image.read()
            base64_image = base64.b64encode(content).decode('utf-8')
            image_inputs.append(ImageInput(
                image_base64=base64_image,
                imageid=f"{image.filename}_{idx}"
            ))

        # Initialize analyzer
        analyzer = OpenAIImageAnalyzer(api_key=api_key)

        # Set analysis parameters aligned with image_analysis.py
        openai_params = OpenAIParams(
            max_tokens=max_tokens,
            detail=detail_level,
            model=model,
            temperature=0.2,  # Aligned with image_analysis.py
            presence_penalty=0.1,  # Aligned with image_analysis.py
            frequency_penalty=0.1  # Aligned with image_analysis.py
        )

        # Adjust manipulation parameters based on number of images, aligned with image_analysis.py
        manipulation_params = ImageManipulationParams(
            quality_ratio=0.4,  # Aligned with image_analysis.py
            resize_ratio=0.15 if len(image_inputs) == 1 else 0.25,  # Aligned with image_analysis.py
            max_width=384 if len(image_inputs) == 1 else 576,  # Aligned with image_analysis.py
            max_height=384 if len(image_inputs) == 1 else 576  # Aligned with image_analysis.py
        )

        # Process analysis
        result, metrics = await analyzer.process_request(
            images=image_inputs,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            event_level=event_level,
            openai_params=openai_params,
            manipulation_params=manipulation_params
        )

        # Prepare response
        response = {
            "type": "event" if event_level else "individual",
            "time_taken": metrics["total_time_taken"],
            "token_usage": metrics["token_usage"],
            "cost": metrics.get("cost", {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}),
            "model": metrics["model"]
        }

        if event_level:
            response.update({
                "event_insights": result.event_insights,
                "image_analyses": [
                    {
                        "imageid": analysis.imageid,
                        "quality_score": analysis.quality_score,
                        "quality_metrics": analysis.quality_metrics,
                        "exif_data": {
                            "datetime": analysis.exif_data.datetime,
                            "gps_latitude": analysis.exif_data.gps_latitude,
                            "gps_longitude": analysis.exif_data.gps_longitude,
                            "camera_make": analysis.exif_data.camera_make,
                            "camera_model": analysis.exif_data.camera_model
                        },
                        "analysis": analysis.analysis
                    } for analysis in result.image_analyses
                ],
                "total_images": result.total_images,
                "average_quality": result.average_quality
            })
        else:
            response.update({
                "analyses": [
                    {
                        "imageid": analysis.imageid,
                        "quality_score": analysis.quality_score,
                        "quality_metrics": analysis.quality_metrics,
                        "exif_data": {
                            "datetime": analysis.exif_data.datetime,
                            "gps_latitude": analysis.exif_data.gps_latitude,
                            "gps_longitude": analysis.exif_data.gps_longitude,
                            "camera_make": analysis.exif_data.camera_make,
                            "camera_model": analysis.exif_data.camera_model
                        },
                        "analysis": analysis.analysis
                    } for analysis in result
                ]
            })

        logger.info(f"Analysis completed in {metrics['total_time_taken']:.2f}s, Tokens: {metrics['token_usage']['total_tokens']}, Cost: ${metrics['cost']['total_cost']:.6f}")
        return JSONResponse(content=response)

    except httpx.TimeoutException as e:
        logger.error(f"Timeout during analysis: {str(e)}")
        raise HTTPException(status_code=504, detail="OpenAI API timed out.")
    except httpx.RequestError as e:
        logger.error(f"Network error during analysis: {str(e)}")
        raise HTTPException(status_code=503, detail="Network error occurred.")
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)