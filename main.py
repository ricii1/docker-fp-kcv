from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from utils import load_model, generate_image

app = FastAPI()
model = load_model()

@app.get("/generate")
def generate():
    image_stream = generate_image(model, steps=5, alpha=1.0)
    return StreamingResponse(image_stream, media_type="image/png")

@app.get("/ping")
def ping():
    return {"status": "pong"}