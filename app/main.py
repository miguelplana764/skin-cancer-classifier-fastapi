# app/main.py

import shutil
import os
import uuid
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.predict import predict_image

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

os.makedirs("app/uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="app/uploads"), name="uploads")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/model-info", response_class=HTMLResponse)
async def model_info(request: Request):
    return templates.TemplateResponse("model_info.html", {"request": request})

@app.post("/upload-image", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):

    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = f"app/uploads/{unique_filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return templates.TemplateResponse("main.html", {
        "request": request,
        "image_path": f"/uploads/{unique_filename}"
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image_path: str = Form(...)):
    label, confidence, all_predictions = predict_image(f"app{image_path}")
    confidence_percent = f"{confidence * 100:.2f}%"

    return templates.TemplateResponse("main.html", {
        "request": request,
        "image_path": image_path,
        "label": label,
        "confidence": confidence_percent,
        "predictions": all_predictions
    })
