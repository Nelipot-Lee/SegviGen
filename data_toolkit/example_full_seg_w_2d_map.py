import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from trellis2 import models
from color_glb import color_glb
from color_img import color_img
from glb_to_vxz import glb_to_vxz
from vxz_to_slat import vxz_to_slat
from img_to_cond import img_to_cond
from glb_to_parts import glb_to_parts
from trellis2.pipelines.rembg import BiRefNet
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

rembg_model = BiRefNet(model_name="briaai/RMBG-2.0")
rembg_model.cuda()
image_cond_model = DinoV3FeatureExtractor(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
image_cond_model.cuda()

shape_encoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16").cuda().eval()
tex_encoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16").cuda().eval()

glb = "./assets/example.glb"
input_vxz = "./assets/input.vxz"
parts_path = "./assets/parts"
full_seg_w_2d_map_path = "./assets/full_seg_w_2d_map"
interactive = False

transforms = "transforms.json"
img = "./assets/full_seg_w_2d_map/2d_map.png"
cond = "./assets/full_seg_w_2d_map/cond.pth"

colors = os.path.join(full_seg_w_2d_map_path, "colors.json")
output_glb_path = os.path.join(full_seg_w_2d_map_path, "output.glb")
output_vxz_path = os.path.join(full_seg_w_2d_map_path, "output.vxz")

glb_to_vxz(glb, input_vxz)
glb_to_parts(glb, parts_path)
color_glb(parts_path, full_seg_w_2d_map_path, interactive)

color_img(parts_path, img, transforms, colors)
img_to_cond(rembg_model, image_cond_model, img, cond)
glb_to_vxz(output_glb_path, output_vxz_path)
vxz_to_slat(shape_encoder, tex_encoder, input_vxz, output_vxz_path, full_seg_w_2d_map_path, interactive)