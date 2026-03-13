from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")

# Serve CSS & frontend
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    results = model(image)[0]
    name = results.names[results.probs.top1]
    conf = results.probs.top1conf.item() * 100
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": name,
            "confidence": f"{conf:.2f}%",
        },
    )