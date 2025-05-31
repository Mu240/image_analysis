import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API and Model Configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

AVAILABLE_MODELS = ["gpt-4.1", "gpt-4o", "gpt-4-turbo"]
DEFAULT_MODEL = "gpt-4.1"
VISION_DETAIL_LEVEL = "low"
DEFAULT_MAX_TOKENS = 600

# File and Processing Limits
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 10))  # Default to 10MB if not set
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_FILE_TYPES = {"image/jpeg", "image/png"}
MAX_MEMORY_USAGE_MB = int(os.getenv("MAX_MEMORY_USAGE_MB", 512))  # Default to 512MB
PROCESSING_TIMEOUT_SECONDS = int(os.getenv("PROCESSING_TIMEOUT_SECONDS", 300))  # Default to 300s

# Image Processing Parameters
QUALITY_RATIO = 0.4
RESIZE_RATIO_SINGLE = 0.15
RESIZE_RATIO_MULTIPLE = 0.25
MAX_WIDTH_SINGLE = 384
MAX_HEIGHT_SINGLE = 384
MAX_WIDTH_MULTIPLE = 576
MAX_HEIGHT_MULTIPLE = 576

# OpenAI Parameters
TEMPERATURE = 0.2
PRESENCE_PENALTY = 0.1
FREQUENCY_PENALTY = 0.1