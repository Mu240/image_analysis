from dotenv import load_dotenv
import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import base64
import logging
from typing import List, Dict, Any
from pydantic import BaseModel
from image_analysis import OpenAIImageAnalyzer, ImageInput, OpenAIParams, ImageManipulationParams, CollectiveAnalysis
import os
import httpx
import psutil
from config import (
    OPENAI_API_KEY, AVAILABLE_MODELS, DEFAULT_MODEL, VISION_DETAIL_LEVEL, DEFAULT_MAX_TOKENS,
    MAX_FILE_SIZE_BYTES, ALLOWED_FILE_TYPES, MAX_MEMORY_USAGE_MB, PROCESSING_TIMEOUT_SECONDS,
    QUALITY_RATIO, RESIZE_RATIO_SINGLE, RESIZE_RATIO_MULTIPLE, MAX_WIDTH_SINGLE, MAX_HEIGHT_SINGLE,
    MAX_WIDTH_MULTIPLE, MAX_HEIGHT_MULTIPLE, TEMPERATURE, PRESENCE_PENALTY, FREQUENCY_PENALTY
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OpenAI Image Analysis API",
    description="API for analyzing images using OpenAI GPT-4 Vision, supporting event-level and individual analysis.",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request validation
class AnalysisRequest(BaseModel):
    user_prompt: str
    system_prompt: str
    detail_level: str = VISION_DETAIL_LEVEL
    event_level: bool = True
    model: str = DEFAULT_MODEL

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "OpenAI Image Analysis API is running", "status": "healthy"}

@app.get("/config")
async def get_config():
    """Return configuration values for the front-end"""
    analyzer = OpenAIImageAnalyzer(api_key=OPENAI_API_KEY)
    return {
        "max_tokens": DEFAULT_MAX_TOKENS,
        "api_key_status": analyzer.get_api_key_status()
    }

@app.post("/analyze")
async def analyze_images(
    images: List[UploadFile] = File(...),
    user_prompt: str = Form(...),
    system_prompt: str = Form(...),
    detail_level: str = Form(VISION_DETAIL_LEVEL),
    event_level: bool = Form(True),
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
        if model not in AVAILABLE_MODELS:
            logger.warning(f"Invalid model {model}, using default {DEFAULT_MODEL}")
            model = DEFAULT_MODEL
        if not user_prompt or not system_prompt:
            raise HTTPException(status_code=400, detail="User prompt and system prompt are required")

        # Validate file types and sizes
        image_inputs = []
        for image in images:
            if image.content_type not in ALLOWED_FILE_TYPES:
                raise HTTPException(status_code=400, detail=f"File {image.filename} must be JPEG or PNG")
            content = await image.read()
            if len(content) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {image.filename} exceeds maximum size of {MAX_FILE_SIZE_BYTES / 1024 / 1024}MB"
                )
            base64_image = base64.b64encode(content).decode('utf-8')
            image_inputs.append(ImageInput(image_base64=base64_image, imageid=f"{image.filename}_{len(image_inputs)}"))
            # Reset file pointer and clean up
            await image.seek(0)

        # Check memory usage
        process = psutil.Process(os.getpid())
        mem_usage_mb = process.memory_info().rss / 1024 / 1024
        if mem_usage_mb > MAX_MEMORY_USAGE_MB:
            raise HTTPException(
                status_code=503,
                detail=f"Server memory usage exceeds limit of {MAX_MEMORY_USAGE_MB}MB: {mem_usage_mb:.2f}MB"
            )

        # Initialize analyzer
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
        analyzer = OpenAIImageAnalyzer(api_key=OPENAI_API_KEY)

        # Set analysis parameters
        openai_params = OpenAIParams(
            max_tokens=DEFAULT_MAX_TOKENS,
            detail=detail_level,
            model=model,
            temperature=TEMPERATURE,
            presence_penalty=PRESENCE_PENALTY,
            frequency_penalty=FREQUENCY_PENALTY
        )

        # Adjust manipulation parameters based on number of images
        manipulation_params = ImageManipulationParams(
            quality_ratio=QUALITY_RATIO,
            resize_ratio=RESIZE_RATIO_SINGLE if len(image_inputs) == 1 else RESIZE_RATIO_MULTIPLE,
            max_width=MAX_WIDTH_SINGLE if len(image_inputs) == 1 else MAX_WIDTH_MULTIPLE,
            max_height=MAX_HEIGHT_SINGLE if len(image_inputs) == 1 else MAX_HEIGHT_MULTIPLE
        )

        # Process analysis with timeout
        try:
            result, metrics = await asyncio.wait_for(
                analyzer.process_request(
                    images=image_inputs,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    event_level=event_level,
                    openai_params=openai_params,
                    manipulation_params=manipulation_params
                ),
                timeout=PROCESSING_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"Processing timed out after {PROCESSING_TIMEOUT_SECONDS} seconds")
        except Exception as e:
            logger.error(f"Error during analysis processing: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal processing error")

        # Prepare response
        response = {
            "type": "event" if event_level else "individual",
            "time_taken": metrics.get("total_time_taken", round(time.time() - start_time, 2)),
            "token_usage": metrics.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            "cost": metrics.get("cost", {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}),
            "model": metrics.get("model", DEFAULT_MODEL)
        }

        if event_level and isinstance(result, CollectiveAnalysis):
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
        elif not event_level and isinstance(result, list):
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

        logger.info(f"Analysis completed in {response['time_taken']:.2f}s, Tokens: {response['token_usage']['total_tokens']}, Cost: ${response['cost']['total_cost']:.6f}")
        return JSONResponse(content=response)

    except httpx.TimeoutException as e:
        logger.error(f"Timeout during OpenAI API call: {str(e)}")
        raise HTTPException(status_code=504, detail="OpenAI API timed out")
    except httpx.RequestError as e:
        logger.error(f"Network error during OpenAI API call: {str(e)}")
        raise HTTPException(status_code=503, detail="Network error occurred")
    except HTTPException as e:
        logger.error(f"HTTP exception: {str(e.detail)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
    finally:
        # Ensure all uploaded files are closed
        for image in images:
            await image.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
