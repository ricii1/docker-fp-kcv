from model import StyleGAN
import torch
from io import BytesIO
from torchvision.utils import save_image
import numpy as np
import legacy
from PIL import Image
import time

LATENT_FEATURES = 512
RESOLUTION = 128

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_model_pt(path='model_128.pt'):
    model = StyleGAN(LATENT_FEATURES, RESOLUTION).to(DEVICE)
    last_checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(last_checkpoint['generator'], strict=False)
    model.eval()
    return model

def generate_image_stylegan(generator, steps=5, alpha=1.0):
    with torch.no_grad():
        image = generator(torch.randn(1, LATENT_FEATURES, device=DEVICE), alpha=1.0, steps=steps)
        image = image.tanh()
        image = (image + 1) / 2 

        buffer = BytesIO()
        save_image(image, buffer, format='PNG')
        buffer.seek(0)
        return buffer
    
def load_model_pkl(path='styleganv2.pkl'):
    with open(path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE)
    G.eval()
    return G

def generate_image_from_pkl(generator, seed=0, trunc=1):
    start = time.time()
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, generator.z_dim)).to(DEVICE)
    label = torch.zeros(1, generator.c_dim, device=DEVICE)
    img = generator(z, label, truncation_psi=trunc, noise_mode='const')
    img = (img + 1) * (255 / 2)
    img = img.clamp(0, 255).to(torch.uint8)
    img = img[0].permute(1, 2, 0).cpu().numpy() # (Channel, Height, Width) to (Height, Width, Channel)
    pil_image = Image.fromarray(img)

    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)

    end = time.time()
    print(f"Image generation time: {end - start:.2f} seconds")

    return buffer
