# AI Image Analysis Studio

This project provides an API and frontend for analyzing images using OpenAI's GPT-4 Vision model. It supports both event-level and individual image analysis, extracting metadata such as event details, quality scores, and EXIF data.

## Setup Instructions

### Install Dependencies
Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure Environment
Create a `.env` file in the project root with the following:

```
OPENAI_API_KEY="your-openai-api-key"
MAX_FILE_SIZE_MB=10
MAX_MEMORY_USAGE_MB=512
PROCESSING_TIMEOUT_SECONDS=300
```

### Run the Backend
Start the FastAPI server:

```bash
# In the directory: D:\advancedimage\backend>
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Serve the Frontend
Place `index.html` in a web server directory and serve it (e.g., using a simple HTTP server):

```bash
# In the directory: D:\advancedimage\frontend>
python -m http.server 8080
```

### Access the Application
- Open `http://localhost:8080` in your browser to access the frontend.
- Ensure the backend is running at `http://localhost:8000`.

## API Documentation

The backend provides a REST API built with FastAPI for image analysis.

### Endpoints
- **GET `/`**
  - Health check endpoint.
  - Returns: `{"message": "OpenAI Image Analysis API is running"}`

- **GET `/config`**
  - Returns configuration values for the frontend, including API key status and max tokens.
  - Returns: JSON object with configuration details.

- **POST `/analyze`**
  - Analyzes uploaded images using OpenAI GPT-4 Vision.
  - Parameters:
    - `images`: List of image files (max 10, JPEG/PNG, max size 10MB each)
    - `user_prompt`: String (required)
    - `system_prompt`: String (required)
    - `detail_level`: String (default: "low")
    - `event_level`: Boolean (default: true)
    - `model`: String (default: "gpt-4.1", options: "gpt-4.1", "gpt-4o", "gpt-4-turbo")
  - Returns: JSON object with analysis results (event-level or individual).

### Swagger UI
FastAPI provides an interactive API documentation interface. Once the backend is running:
- Open `http://localhost:8000/docs` to access the Swagger UI.
- Use the interface to test the API endpoints and view detailed schema information.

## Security Notes
- **File Validation**: Only JPEG and PNG files are allowed, with a maximum size of 10MB per file (configurable in `.env`).
- **Memory Limits**: The server enforces a memory usage limit (default: 512MB, configurable in `.env`).
- **Timeouts**: Processing is limited to 300 seconds (configurable in `.env`) to prevent abuse.
- **File Cleanup**: Uploaded files are properly closed and cleaned up after processing.

## Project Structure
- `backend/`
  - `server.py`: FastAPI backend server.
  - `image_analysis.py`: Core image analysis logic.
  - `config.py`: Configuration constants and environment variable handling.
- `frontend/`
  - `index.html`: Frontend interface for uploading and analyzing images.
- `.env`: Environment variables (not tracked in git).
- `.gitignore`: Git ignore rules.
- `requirements.txt`: Python dependencies.

## Additional Notes
- Ensure you have Python 3.8+ installed.
- The project uses OpenAI's GPT-4 Vision models, so a valid API key is required.
- For production, adjust the CORS settings in `server.py` to allow specific origins instead of `"*"`.
- Logs are generated for debugging; check the console for detailed information during development.
