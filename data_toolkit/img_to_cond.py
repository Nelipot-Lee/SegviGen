import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import numpy as np

from PIL import Image
from trellis2.pipelines.rembg import BiRefNet
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

def preprocess_image(rembg_model, input):
    if input.mode != "RGB":
        bg = Image.new("RGB", input.size, (255, 255, 255))
        bg.paste(input, mask=input.split()[3])
        input = bg
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(input.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
    if has_alpha:
        output = input
    else:
        input = input.convert('RGB')
        output = rembg_model(input)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output

def get_cond(image_cond_model, image):
    image_cond_model.image_size = 512
    cond = image_cond_model(image)
    neg_cond = torch.zeros_like(cond)
    return {'cond': cond.cpu(), 'neg_cond': neg_cond.cpu()}

def img_to_cond(rembg_model, image_cond_model, image_path, save_path):
    image = Image.open(image_path)
    image = preprocess_image(rembg_model, image)
    cond_dict = get_cond(image_cond_model, [image])
    torch.save(cond_dict, save_path)

if __name__ == "__main__":
    image_path = "./assets/img.png"
    save_path = "./assets/cond.pth"
    rembg_model = BiRefNet(model_name="briaai/RMBG-2.0")
    rembg_model.cuda()
    image_cond_model = DinoV3FeatureExtractor(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
    image_cond_model.cuda()
    img_to_cond(rembg_model, image_cond_model, image_path, save_path)