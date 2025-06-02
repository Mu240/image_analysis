import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API and Model Configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Custom API key for request authentication
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in .env file")

AVAILABLE_MODELS = ["gpt-4.1", "gpt-4o", "gpt-4-turbo"]
DEFAULT_MODEL = "gpt-4.1"
VISION_DETAIL_LEVEL = "low"

# Validate MAX_TOKENS
try:
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 600))
    if not (100 <= MAX_TOKENS <= 4000):
        raise ValueError("MAX_TOKENS must be between 100 and 4000")
except ValueError as e:
    raise ValueError(f"Invalid MAX_TOKENS in .env file: {str(e)}")

# File and Processing Limits
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 10))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_FILE_TYPES = {"image/jpeg", "image/png"}
MAX_MEMORY_USAGE_MB = int(os.getenv("MAX_MEMORY_USAGE_MB", 512))
PROCESSING_TIMEOUT_SECONDS = int(os.getenv("PROCESSING_TIMEOUT_SECONDS", 300))

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
