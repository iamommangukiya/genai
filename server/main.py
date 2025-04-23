import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil

app = FastAPI()

# Path where uploaded files will be stored
UPLOAD_DIR = "uploaded_files"

# Make sure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)
app = FastAPI()

# Route for the home page
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Route for a path parameter
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

# Route for POST request
@app.post("/items/")
def create_item(item: dict):
    return {"message": "Item created", "item": item}

# Endpoint to upload PDF files
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Check if the uploaded file is a PDF
    if file.content_type != "application/pdf":
        return JSONResponse(content={"error": "File must be a PDF"}, status_code=400)

    # Construct the full file path
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    # Check if file already exists
    if os.path.exists(file_location):
        # If file exists, delete it
        os.remove(file_location)

    # Save the uploaded file
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"filename": file.filename, "location": file_location}



# Run the application with Uvicorn:
# uvicorn main:app --reload
