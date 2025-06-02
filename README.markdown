# AI Image Analysis Platform

The AI Image Analysis Platform is a web application that leverages OpenAI's GPT-4 Vision API to analyze images, offering both event-level and individual image analysis. It includes advanced features such as image manipulation metrics, EXIF metadata extraction, quality scoring, and configurable analysis parameters, with a Flask-based backend and server-side rendered templates.

## Features

- **Event-Level Analysis**: Aggregates insights across multiple images to identify event details (e.g., name, location, activities) using EXIF GPS data or visual inference.
- **Individual Image Analysis**: Provides detailed per-image analysis, including quality scores, EXIF metadata, and AI-generated insights.
- **Image Manipulation Metrics**: Reports size reduction percentage (based on pixel count) and quality reduction percentage (based on compression) for each image, displayed in the API response.
- **Configurable Parameters**:
  - `max_tokens`: Adjustable token limit for OpenAI API calls (100–4000, default: 600).
  - Model selection: GPT-4.1, GPT-4o, GPT-4-turbo.
  - Detail level: low, medium, high (auto-adjusts based on image count).
- **Strict File Validation**: Accepts only JPEG and PNG images, with a 10MB size limit per file and a maximum of 10 images per request.
- **EXIF Metadata Extraction**: Extracts comprehensive metadata (datetime, GPS coordinates, camera details) using PIL, piexif, and exifread for reliability.
- **Image Quality Scoring**: Computes scores based on brightness, contrast, sharpness, and noise, with detailed metrics.
- **Flask Backend**: Built with Flask, supporting server-side rendering with templates, session management, and detailed logging.
- **API Key Authentication**: Requires a valid API key in the `Authorization` header for the `/analyze` endpoint.
- **Resource Management**: Enforces memory limits (512MB), processing timeouts (300 seconds), and proper file cleanup.

## Requirements

- Python 3.8+
- OpenAI API key with GPT-4 Vision access
- Dependencies listed in `requirements.txt`
- `.env` file with environment variables, including `API_KEY` for request authentication

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mu240/image_analysis.git
   cd image_analysis
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your-openai-api-key
   API_KEY=your-custom-api-key
   MAX_FILE_SIZE_MB=10
   MAX_MEMORY_USAGE_MB=512
   PROCESSING_TIMEOUT_SECONDS=300
   MAX_TOKENS=600
   ```
   Replace `your-openai-api-key` with your OpenAI API key and `your-custom-api-key` with a secure API key for request authentication. Adjust `MAX_TOKENS` as needed (100–4000).

5. **Run the Backend**:
   Navigate to the `backend/` directory and start the FastAPI server:
   ```bash
   cd backend
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

6. **Serve the Frontend**:
   Navigate to the `frontend/` directory and serve `index.html` using a simple HTTP server:
   ```bash
   cd frontend
   python -m http.server 8080
   ```
   Alternatively, place `index.html` in a web server directory (e.g., Nginx, Apache).

7. **Access the Application**:
   - Open `http://localhost:8080` in a browser to access the frontend.
   - Ensure the backend is running at `http://localhost:8000`.

## Usage

1. **Upload Images**:
   - Navigate to the image upload route (e.g., `/upload`, depending on `main.py` implementation).
   - Upload up to 10 JPEG or PNG images.
   - Maximum file size per image is 10MB (configurable via `MAX_FILE_SIZE_MB`).

2. **Configure Analysis Settings**:
   - **Analysis Mode**: Choose between event-level or individual analysis (via form or API).
   - **User Prompt**: Provide a prompt to guide the AI (e.g., "Analyze these images for a wedding event").
   - **System Prompt**: Define the JSON output structure and analysis rules.
   - **Detail Level**: Select "low", "medium", or "high".
   - **Model**: Choose from GPT-4.1, GPT-4o, or GPT-4-turbo.
   - **Maximum Tokens**: Set a value between 100 and 4000 (default: 600).

3. **Analyze Images**:
   - Submit the form or send an API request to the `/analyze` endpoint.
   - The application processes the images and displays results (rendered via templates).

4. **View Results**:
   - Results are displayed via server-side rendered templates (e.g., `/results` route).
   - Includes analysis type, time taken, tokens used, cost, model, and detailed per-image or event-level insights.

## API Documentation

The backend provides a REST API for programmatic access. The API endpoints are defined in `main.py`.

### Endpoints

#### `GET /`
- **Description**: Health check endpoint (if implemented in `main.py`).
- **Returns**:
  ```json
  {"message": "AI Image Analysis API is running", "status": "healthy"}
  ```

#### `GET /config`
- **Description**: Returns configuration values (e.g., `max_tokens`).
- **Returns**:
  ```json
  {
    "max_tokens": 600,
    "api_key_status": {
      "api_key_set": true,
      "api_key_prefix": "sk-p"
    }
  }
  ```

#### `POST /analyze`
- **Description**: Analyzes uploaded images using OpenAI GPT-4 Vision.
- **Headers**:
  - `Authorization`: `Bearer <your-custom-api-key>` (required, must match `API_KEY` in `.env`)
- **Parameters**:
  - `images`: List of image files (max 10, JPEG/PNG, max 10MB each)
  - `user_prompt`: String (required, max 2000 characters)
  - `system_prompt`: String (required, max 2000 characters)
  - `detail_level`: String (default: "low", options: "low", "medium", "high")
  - `analysis_mode`: String (default: "event_level", options: "event_level", "individual_level")
  - `model`: String (default: "gpt-4.1", options: "gpt-4.1", "gpt-4o", "gpt-4-turbo")
  - `max_tokens`: Integer (default: 600, range: 100–4000)
- **Returns**: JSON object with analysis results (event-level or individual), including:
  - `type`: "event" or "individual"
  - `time_taken`: Processing time in seconds
  - `token_usage`: Prompt, completion, and total tokens
  - `cost`: Input, output, and total cost
  - `model`: Model used
  - `manipulation_metrics`: Size and quality reduction percentages per image
  - Event-level: `event_insights`, `image_analyses`, `total_images`, `average_quality`
  - Individual: `analyses`
- **Example Request**:
  ```bash
  curl -X POST "http://localhost:5000/analyze" \
    -H "Authorization: Bearer your-custom-api-key" \
    -F "images=@image1.jpg" \
    -F "user_prompt=Analyze these images for a wedding event" \
    -F "system_prompt=Generate batch-level metadata for a wedding event..." \
    -F "detail_level=low" \
    -F "analysis_mode=event_level" \
    -F "model=gpt-4.1" \
    -F "max_tokens=600"
  ```
- **Example Response (Event-Level)**:
  ```json
  {
    "type": "event",
    "time_taken": 5.23,
    "token_usage": {
      "prompt_tokens": 340,
      "completion_tokens": 300,
      "total_tokens": 640
    },
    "cost": {
      "input_cost": 0.0017,
      "output_cost": 0.0045,
      "total_cost": 0.0062
    },
    "model": "gpt-4.1",
    "manipulation_metrics": {
      "image1.jpg_0": {
        "size_reduction_percent": 97.75,
        "quality_reduction_percent": 60.0
      }
    },
    "event_insights": {
      "eventType": "wedding",
      "eventMetadata": {
        "eventName": "Smith Wedding",
        "eventSubtype": "Ceremony",
        "eventTheme": "Rustic",
        "eventTitle": "Smith & Johnson Wedding Ceremony"
      },
      "eventLocation": {
        "name": "Rustic Barn Venue",
        "type": "venue",
        "address": "123 Country Rd, Springfield"
      },
      "eventActivities": [
        {
          "name": "Vow Exchange",
          "estimatedTime": "3:00 PM",
          "activityLocation": {
            "name": "Rustic Barn Venue",
            "type": "venue",
            "address": "123 Country Rd, Springfield"
          },
          "imageid": "image1.jpg_0"
        }
      ]
    },
    "image_analyses": [
      {
        "imageid": "image1.jpg_0",
        "quality_score": 75.0,
        "quality_metrics": {
          "overall_quality": "75.00/100",
          "brightness": "128.00/255",
          "contrast": "50.00",
          "sharpness": "80.00/100",
          "noise_estimate": "20.00/100",
          "quality_category": "Good"
        },
        "manipulation_metrics": {
          "size_reduction_percent": 97.75,
          "quality_reduction_percent": 60.0
        },
        "exif_data": {
          "datetime": "2023:10:15 15:00:00",
          "gps_latitude": 39.7392,
          "gps_longitude": -104.9903,
          "camera_make": "Canon",
          "camera_model": "EOS R5"
        },
        "analysis": {
          "description": "Vow Exchange",
          "location": {
            "name": "Rustic Barn Venue",
            "type": "venue",
            "address": "123 Country Rd, Springfield"
          },
          "estimatedTime": "3:00 PM",
          "qualityScore": 75.0
        }
      }
    ],
    "total_images": 1,
    "average_quality": 75.0
  }
  ```

## Security Notes

- **API Key Authentication**: The `/analyze` endpoint requires a valid `API_KEY` in the `Authorization` header (`Bearer <your-custom-api-key>`). Set this in `.env` and do not share it publicly.
- **File Validation**: Only JPEG and PNG files are accepted, with a 10MB size limit per file (configurable via `MAX_FILE_SIZE_MB`).
- **Memory Limits**: Enforces a 512MB memory usage limit (configurable via `MAX_MEMORY_USAGE_MB`).
- **Timeouts**: Processing is capped at 300 seconds (configurable via `PROCESSING_TIMEOUT_SECONDS`).
- **File Cleanup**: Uploaded files are closed and cleaned up after processing to prevent resource leaks.
- **CORS**: Configure CORS in `main.py` if needed for cross-origin requests (not enabled by default in Flask).
- **API Key Security**: Store both `OPENAI_API_KEY` and `API_KEY` in `.env` and ensure `.env` is not committed to version control (listed in `.gitignore`).

## Project Structure

```
image_analysis/
├── backend/
│   ├── config.py          # Configuration settings and environment variable handling
│   ├── image_analysis.py  # Core image processing and analysis logic
│   └── server.py          # FastAPI backend server
├── frontend/
│   └── index.html         # Frontend interface
├── .env                   # Environment variables (not tracked)
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## Troubleshooting

- **CORS Errors**: Flask does not enable CORS by default. If needed, install `flask-cors` and configure it in `main.py`.
- **API Key Errors**: Verify both `OPENAI_API_KEY` and `API_KEY` in `.env` are valid. For `/analyze`, include the correct `API_KEY` in the `Authorization` header. Check logs for details.
- **File Upload Issues**: Confirm images are JPEG or PNG and under 10MB. Check server logs for errors.
- **Slow Processing**: Increase `PROCESSING_TIMEOUT_SECONDS` or `MAX_MEMORY_USAGE_MB` in `.env` for larger images.
- **Template Not Found Errors**: Ensure templates are in the `templates/` folder and correctly referenced in `main.py`.

## Additional Notes

- **Python Version**: Requires Python 3.8 or higher.
- **OpenAI API**: A valid API key with GPT-4 Vision access is required.
- **Custom API Key**: The `API_KEY` is used for request authentication and must be included in the `Authorization` header for `/analyze` requests.
- **Image Manipulation Metrics**: Size reduction is calculated as the percentage decrease in pixel count; quality reduction is based on the compression ratio (e.g., `QUALITY_RATIO=0.4` yields 60% reduction).
- **EXIF Extraction**: Uses multiple libraries (PIL, piexif, exifread) for robust metadata extraction, including GPS-based address resolution.
- **Quality Scoring**: Combines brightness, contrast, sharpness, and noise metrics for a comprehensive score.
- **Production Considerations**:
  - Configure CORS in `main.py` if needed.
  - Increase `MAX_MEMORY_USAGE_MB` or `PROCESSING_TIMEOUT_SECONDS` for larger workloads.
  - Use a production WSGI server (e.g., Gunicorn) with Flask for deployment.
- **Logging**: Logs are written to `music_analysis.log`; check this file for errors or performance metrics.
- **Dependencies**: Key libraries include `flask`, `openai`, `pillow`, `piexif`, `exifread`, `geopy`, `python-dotenv`, and `psutil`.

## Contributing

Contributions are welcome! Submit issues or pull requests to the repository. Follow PEP 8 for code style and include tests for new features.

## License

MIT License (add a `LICENSE` file to the project if needed).

## Last Updated

09:48 PM PKT on Monday, June 2, 2025
