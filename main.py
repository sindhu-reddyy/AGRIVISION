from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference.predict import load_resources, predict_plant_disease
import io
import uvicorn
import shutil
import os

app = FastAPI(title="AgriVision API", description="AI Plant Diagnosis Backend")

# CORS
origins = [
    "http://localhost:5173",  # Vite default
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    try:
        load_resources()
    except Exception as e:
        print(f"Failed to load resources: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to AgriVision AI Forest"}

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        contents = await image.read()
        image_stream = io.BytesIO(contents)
        
        # Run inference
        print(f"📩 Received request for question: '{question}'")
        answer = predict_plant_disease(image_stream, question)
        print(f"📤 Sending response: '{answer}'")
        
        return {"answer": answer}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
