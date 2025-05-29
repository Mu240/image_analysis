Install Dependencies:Create a virtual environment and install required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Run the Backend:Start the FastAPI server:
in the directory like:
D:\advancedimage\backend>
uvicorn server:app --host 0.0.0.0 --port 8000


Serve the Frontend which is in frontend folder run cmd then run below command:

Place index.html in a web server directory (e.g., using a simple HTTP server) in the directory like:
D:\advancedimage\frontend>
python -m http.server 8080


Access the Application:

Open http://localhost:8080 in your browser.
Ensure the backend is running at http://localhost:8000.

and put openai key on webpage