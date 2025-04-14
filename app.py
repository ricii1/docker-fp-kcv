from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from utils import load_model_pt, generate_image_stylegan, load_model_pkl, generate_image_from_pkl, generate_image_vanillagan
import random

app = FastAPI()
stylegan = load_model_pt("model_128.pt", model_type="stylegan")
styleganv2 = load_model_pkl("styleganv2.pkl")
gan = load_model_pt("generator_VanillaGAN.pt", model_type="vanillagan")

@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI StyleGAN API"}

@app.get("/ping")
def ping():
    return {"status": "pong"}

@app.get("/generate/stylegan")
def generate_stylegan():
    image_stream = generate_image_stylegan(stylegan, steps=5, alpha=1.0)
    return StreamingResponse(image_stream, media_type="image/png")

@app.get("/generate/styleganv2")
def generate_styleganv2(seed: int = Query(-1)):
    if seed == -1: 
        seed = random.randint(0, 65535)
    image_stream = generate_image_from_pkl(styleganv2, seed=seed, trunc=1)
    return StreamingResponse(image_stream, media_type="image/png")

@app.get("/generate/vanillagan")
def generate_vanillagan():
    image_stream = generate_image_vanillagan(gan)
    return StreamingResponse(image_stream, media_type="image/png")

