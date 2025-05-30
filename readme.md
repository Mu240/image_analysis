## Overview
This project is an AI-powered image analysis studio leveraging OpenAI's GPT-4 Vision capabilities. It allows users to upload images, configure analysis settings, and generate metadata or insights, particularly for events like weddings.

## Directory Structure
- `.idea/`: IntelliJ IDEA project files.
- `venv/`: Virtual environment for Python dependencies.
- `backend/`: Backend server code (e.g., `server.py`, `image_analysis.py`).
- `frontend/`: Frontend HTML/CSS/JavaScript code (e.g., `index.html`).
- `.env`: Environment file for API keys (e.g., OpenAI API key).
- `.env`: Environment configuration file.
- `requirements.txt`: Python dependencies list.
- `.gitignore`: Git ignore file to exclude unnecessary files.

## Setup
### Install Dependencies
1. Ensure Python 3.10 is installed.
2. Create a virtual environment and activate it:
   - On Unix/Mac: `python -m venv venv` followed by `source venv/bin/activate`
   - On Windows: `python -m venv venv` followed by `venv\Scripts\activate`
3. Install required packages:
   - Run `pip install -r requirements.txt` in the project root.

### Configure OpenAI API Key
1. Obtain your OpenAI API key from the OpenAI website.
2. Add the key to the `.env` file in the project root as `OPENAI_API_KEY=your_api_key_here`.
3. Alternatively, add the API key directly on the webpage:
   - Open `index.html` in the `frontend` folder.
   - Locate the `<script>` section and modify the `ImageAnalysisUI` constructor to include:
     ```javascript
     this.apiBaseUrl = "http://localhost:8000";
     this.apiKey = "your_openai_api_key_here"; // Add your key here
     ```
   - Save the file and proceed.

## How to Run the Application

### Run the Backend
1. Navigate to the `backend` directory (e.g., `D:\advancedimage\backend`).
2. Start the FastAPI server:
   - Run `uvicorn server:app --host 0.0.0.0 --port 8000`.

### Serve the Frontend
1. Navigate to the `frontend` directory (e.g., `D:\advancedimage\frontend`).
2. Start a simple HTTP server:
   - Run `python -m http.server 8080`.
3. Ensure the server is active (check for output confirming it’s serving on `http://localhost:8080`).

### Access the Application
- Open `http://localhost:8080` in your browser.
- Ensure the backend is running at `http://localhost:8000` for API calls to work.

## Usage
- **Upload Images**: Drag and drop or browse up to 10 JPG/PNG images.
- **Configure Settings**: Adjust user/system prompts, detail level, and model (e.g., GPT-4.1).
- **Analyze**: Click "Analyze" to process images and view results (overview, individual, insights, JSON).

## Features
- Event-level or individual image analysis.
- Quality scoring based on brightness, contrast, sharpness, and noise.
- EXIF data extraction (date, GPS, camera info).
- Interactive UI with drag-and-drop and toggle settings.

