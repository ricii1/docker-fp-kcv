from model import StyleGAN
import torch
from io import BytesIO
from torchvision.utils import save_image

LATENT_FEATURES = 512
RESOLUTION = 128

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_model(path='model_128.pt'):
    model = StyleGAN(LATENT_FEATURES, RESOLUTION).to(DEVICE)
    last_checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(last_checkpoint['generator'], strict=False)
    model.eval()
    return model

def generate_image(generator, steps=5, alpha=1.0):
    with torch.no_grad():
        image = generator(torch.randn(1, LATENT_FEATURES, device=DEVICE), alpha=1.0, steps=steps)
        image = image.tanh()
        image = (image + 1) / 2 

        buffer = BytesIO()
        save_image(image, buffer, format='PNG')
        buffer.seek(0)
        return buffer