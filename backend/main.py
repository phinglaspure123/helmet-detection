# main.py
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import shutil
from typing import Optional

# Import modules
import detection
import database
import email_service
import pdf

app = FastAPI(title="Helmet Detection System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("violations", exist_ok=True)
os.makedirs("challans", exist_ok=True)

# Mount static directories to serve files
app.mount("/violations", StaticFiles(directory="violations"), name="violations")
app.mount("/challans", StaticFiles(directory="challans"), name="challans")

# Initialize detection models when server starts
@app.on_event("startup")
async def startup_event():
    print("Starting up: Initializing detection models...")
    # This ensures models are downloaded and initialized when the server starts
    detection.init_models()
    print("Detection models initialized.")

@app.get("/")
def read_root():
    print("Root endpoint accessed.")
    return {"message": "Helmet Detection API is running"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    print(f"Uploading video: {file.filename}")
    # Save the uploaded video
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"Video saved to {file_path}")
    
    return {"filename": file.filename, "path": file_path}

@app.post("/detect")
async def detect_violations(video_path: str = Form(...)):
    print(f"Starting detection for video: {video_path}")
    
    # Run detection synchronously (this will take time but will complete eventually)
    violations = detection.detect_violations(video_path)
    print(f"Detected {len(violations)} violations.")
    
    # Save violations to database
    for violation in violations:
        print(f"Saving violation ID {violation.get('id')} to database.")
        database.save_violation(violation)
    
    # Send email notifications to each violator
    for violation in violations:
        if violation.get("email"):
            print(f"Sending email to {violation['email']} for violation ID {violation.get('id')}.")
            email_service.send_violation_email(violation["email"], [violation])
    
    # Clean up the temporary video file
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"Temporary video file {video_path} removed.")
    
    # Return the completion response with violation count
    return {
        "status": "completed", 
        "message": "Detection completed successfully",
        "violations_count": len(violations)
    }

@app.get("/violations")
async def get_violations():
    print("Fetching all violations from database.")
    # Get all violations from database
    violations = database.get_all_violations()
    print(f"Retrieved {len(violations)} violations.")
    return {"violations": violations}

@app.get("/generate-challan/{violation_id}")
async def generate_challan(violation_id: str):
    print(f"Generating challan for violation ID: {violation_id}")
    # Get violation details
    violation = database.get_violation_by_id(violation_id)
    if not violation:
        print(f"Violation with ID {violation_id} not found.")
        return JSONResponse(
            status_code=404, 
            content={"message": f"Violation with ID {violation_id} not found"}
        )
    
    # Generate PDF challan
    pdf_path = pdf.generate_challan(violation)
    print(f"Challan generated at {pdf_path}")
    
    # Return the PDF file
    return FileResponse(
        path=pdf_path,
        filename=f"challan_{violation_id}.pdf",
        media_type="application/pdf"
    )

@app.post("/live-stream")
async def process_live_stream(background_tasks: BackgroundTasks):
    print("Starting live stream detection.")
    # This would connect to camera stream and process it
    # For simplicity, we'll just start the detection on a stream
    background_tasks.add_task(detection.process_live_stream)
    print("Live stream detection task added to background.")
    return {"message": "Live stream detection started"}

async def process_video_and_notify(video_path: str):
    print(f"Processing video for violations: {video_path}")
    # Detect violations in video
    violations = detection.detect_violations(video_path)
    print(f"Detected {len(violations)} violations.")
    
    # Save violations to database
    for violation in violations:
        print(f"Saving violation ID {violation.get('id')} to database.")
        database.save_violation(violation)
    
    # Send email notifications to each violator
    for violation in violations:
        if violation.get("email"):
            print(f"Sending email to {violation['email']} for violation ID {violation.get('id')}.")
            email_service.send_violation_email(violation["email"], [violation])
    
    # Clean up the temporary video file
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"Temporary video file {video_path} removed.")

if __name__ == "__main__":
    print("Starting Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 