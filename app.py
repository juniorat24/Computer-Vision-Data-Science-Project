import cv2
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File

# Initialize FastAPI app
app = FastAPI()

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.post("/detect_faces/")
async def detect_faces(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Prepare response
    face_list = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]
    return {"faces_detected": len(faces), "faces": face_list}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
