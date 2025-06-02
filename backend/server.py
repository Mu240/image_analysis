from dotenv import load_dotenv
import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
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
    OPENAI_API_KEY, API_KEY, AVAILABLE_MODELS, DEFAULT_MODEL, VISION_DETAIL_LEVEL, MAX_TOKENS,
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request validation
class AnalysisRequest(BaseModel):
    user_prompt: str
    system_prompt: str
    detail_level: str = VISION_DETAIL_LEVEL
    analysis_mode: str = "event_level"
    model: str = DEFAULT_MODEL
    max_tokens: int = MAX_TOKENS

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "OpenAI Image Analysis API is running", "status": "healthy"}

@app.get("/config")
async def get_config():
    """Return configuration values for the front-end"""
    analyzer = OpenAIImageAnalyzer(api_key=OPENAI_API_KEY)
    return {
        "max_tokens": MAX_TOKENS,
        "api_key_status": analyzer.get_api_key_status()
    }

@app.post("/analyze")
async def analyze_images(
    images: List[UploadFile] = File(...),
    user_prompt: str = Form(...),
    system_prompt: str = Form(...),
    detail_level: str = Form(VISION_DETAIL_LEVEL),
    analysis_mode: str = Form("event_level"),
    model: str = Form(DEFAULT_MODEL),
    max_tokens: int = Form(MAX_TOKENS),
    authorization: str = Header(None)
):
    """Analyze uploaded images using OpenAI GPT-4 Vision with system prompt-defined schema"""
    # Validate API key
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    api_key = authorization[len("Bearer "):].strip()
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        start_time = time.time()
        logger.info(f"Received analysis request: {len(images)} images, analysis_mode={analysis_mode}, model={model}, max_tokens={max_tokens}")

        # Validate images
        if not images:
            raise HTTPException(status_code=400, detail="At least one image is required")
        if len(images) > 10:
            raise HTTPException(status_code=400, detail="Maximum of 10 images allowed")
        for image in images:
            if image.content_type not in ALLOWED_FILE_TYPES:
                raise HTTPException(status_code=400, detail=f"File {image.filename} must be one of {ALLOWED_FILE_TYPES}")
            content = await image.read()
            if len(content) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {image.filename} exceeds maximum size of {MAX_FILE_SIZE_BYTES / 1024 / 1024}MB"
                )
            await image.seek(0)

        # Validate user_prompt
        if not user_prompt:
            raise HTTPException(status_code=400, detail="user_prompt cannot be empty")
        if len(user_prompt) > 2000:
            raise HTTPException(status_code=400, detail="user_prompt cannot exceed 2000 characters")
        if not isinstance(user_prompt, str):
            raise HTTPException(status_code=400, detail="user_prompt must be a string")

        # Validate system_prompt
        if not system_prompt:
            raise HTTPException(status_code=400, detail="system_prompt cannot be empty")
        if len(system_prompt) > 2000:
            raise HTTPException(status_code=400, detail="system_prompt cannot exceed 2000 characters")
        if not isinstance(system_prompt, str):
            raise HTTPException(status_code=400, detail="system_prompt must be a string")

        # Validate detail_level
        allowed_detail_levels = ["low", "medium", "high"]
        if not isinstance(detail_level, str):
            raise HTTPException(status_code=400, detail="detail_level must be a string")
        if detail_level not in allowed_detail_levels:
            raise HTTPException(status_code=400, detail=f"detail_level must be one of {allowed_detail_levels}")

        # Validate analysis_mode
        allowed_analysis_modes = ["event_level", "individual_level"]
        if not isinstance(analysis_mode, str):
            raise HTTPException(status_code=400, detail="analysis_mode must be a string")
        if analysis_mode not in allowed_analysis_modes:
            raise HTTPException(status_code=400, detail=f"analysis_mode must be one of {allowed_analysis_modes}")

        # Validate model
        if not isinstance(model, str):
            raise HTTPException(status_code=400, detail="model must be a string")
        if model not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail=f"model must be one of {AVAILABLE_MODELS}")

        # Validate max_tokens
        if not isinstance(max_tokens, int):
            raise HTTPException(status_code=400, detail="max_tokens must be an integer")
        if not (100 <= max_tokens <= 4000):
            raise HTTPException(status_code=400, detail="max_tokens must be between 100 and 4000")

        # Process images
        image_inputs = []
        for image in images:
            content = await image.read()
            base64_image = base64.b64encode(content).decode('utf-8')
            image_inputs.append(ImageInput(image_base64=base64_image, imageid=f"{image.filename}_{len(image_inputs)}"))
            await image.seek(0)

        # Memory usage check
        process = psutil.Process(os.getpid())
        mem_usage_mb = process.memory_info().rss / 1024 / 1024
        if mem_usage_mb > MAX_MEMORY_USAGE_MB:
            raise HTTPException(
                status_code=503,
                detail=f"Server memory usage exceeds limit of {MAX_MEMORY_USAGE_MB}MB: {mem_usage_mb:.2f}MB"
            )

        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

        analyzer = OpenAIImageAnalyzer(api_key=OPENAI_API_KEY)
        openai_params = OpenAIParams(
            max_tokens=max_tokens,
            detail=detail_level,
            model=model,
            temperature=TEMPERATURE,
            presence_penalty=PRESENCE_PENALTY,
            frequency_penalty=FREQUENCY_PENALTY
        )
        manipulation_params = ImageManipulationParams(
            quality_ratio=QUALITY_RATIO,
            resize_ratio=RESIZE_RATIO_SINGLE if len(image_inputs) == 1 else RESIZE_RATIO_MULTIPLE,
            max_width=MAX_WIDTH_SINGLE if len(image_inputs) == 1 else MAX_WIDTH_MULTIPLE,
            max_height=MAX_HEIGHT_SINGLE if len(image_inputs) == 1 else MAX_HEIGHT_MULTIPLE
        )

        try:
            result, metrics = await asyncio.wait_for(
                analyzer.process_request(
                    images=image_inputs,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    event_level=(analysis_mode == "event_level"),
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

        response = {
            "type": "event" if analysis_mode == "event_level" else "individual",
            "time_taken": metrics.get("total_time_taken", round(time.time() - start_time, 2)),
            "token_usage": metrics.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            "cost": metrics.get("cost", {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}),
            "model": metrics.get("model", DEFAULT_MODEL),
            "manipulation_metrics": metrics.get("manipulation_metrics", {})
        }

        if analysis_mode == "event_level" and isinstance(result, CollectiveAnalysis):
            response.update({
                "event_insights": result.event_insights,
                "image_analyses": [
                    {
                        "imageid": analysis.imageid,
                        "quality_score": analysis.quality_score,
                        "quality_metrics": analysis.quality_metrics,
                        "manipulation_metrics": metrics["manipulation_metrics"].get(analysis.imageid, {}),
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
        elif analysis_mode == "individual_level" and isinstance(result, list):
            response.update({
                "analyses": [
                    {
                        "imageid": analysis.imageid,
                        "quality_score": analysis.quality_score,
                        "quality_metrics": analysis.quality_metrics,
                        "manipulation_metrics": metrics["manipulation_metrics"].get(analysis.imageid, {}),
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
        for image in images:
            await image.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
