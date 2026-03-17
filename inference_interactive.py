import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import trimesh
import o_voxel
import numpy as np
import torch.nn as nn
import trellis2.modules.sparse as sp

from PIL import Image
from tqdm import tqdm
from trellis2 import models
from types import MethodType
from collections import OrderedDict
from torch.nn import functional as F
from trellis2.pipelines.rembg import BiRefNet
from trellis2.modules.utils import manual_cast
from trellis2.representations import MeshWithVoxel
from data_toolkit.bpy_render import render_from_transforms
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

class Sampler:
    def _inference_model(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond):
        t = torch.tensor([t*1000] * x_t.shape[0], dtype=torch.float32).cuda()
        return model(x_t, tex_slat, shape_slat, t, cond, input_points, coords_len_list)

    def guidance_inference_model(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, guidance_strength, guidance_rescale=0.0):
        if guidance_strength == 1:
            return self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['cond'])
        elif guidance_strength == 0:
            return self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['neg_cond'])
        else:
            pred_pos = self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['cond'])
            pred_neg = self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['neg_cond'])
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)
            return pred

    def interval_inference_model(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, sampler_params):
        guidance_strength = sampler_params['guidance_strength']
        guidance_interval = sampler_params['guidance_interval']
        guidance_rescale = sampler_params['guidance_rescale']
        if guidance_interval[0] <= t <= guidance_interval[1]:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, guidance_strength, guidance_rescale)
        else:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, 1, guidance_rescale)

    @torch.no_grad()
    def sample_once(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, t_prev, cond_dict, sampler_params):
        pred_v = self.interval_inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, sampler_params)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return pred_x_prev

    @torch.no_grad()
    def sample(self, model, noise, tex_slat, shape_slat, input_points, coords_len_list, cond_dict, sampler_params):
        sample = noise
        steps = sampler_params['steps']
        rescale_t = sampler_params['rescale_t']
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        for t, t_prev in tqdm(t_pairs, desc="Sampling"):
            sample = self.sample_once(model, sample, tex_slat, shape_slat, input_points, coords_len_list, t, t_prev, cond_dict, sampler_params)
        return sample

def flow_forward(self, x, t, cond, concat_cond, point_embeds, coords_len_list):
    # x.feats: [N, 32]
    x = sp.sparse_cat([x, concat_cond], dim=-1)
    if isinstance(cond, list):
        cond = sp.VarLenTensor.from_tensor_list(cond)
    # x.feats: [N, 64]
    h = self.input_layer(x)
    # h.feats: [N, 1536]
    h = manual_cast(h, self.dtype)
    t_emb = self.t_embedder(t)
    t_emb = self.adaLN_modulation(t_emb)
    t_emb = manual_cast(t_emb, self.dtype)
    cond = manual_cast(cond, self.dtype)
    point_embeds = manual_cast(point_embeds, self.dtype)

    h_feats_list = []
    h_coords_list = []
    begin = 0
    for i, coords_len in enumerate(coords_len_list):
        end = begin + 2 * coords_len
        h_feats_list.append(h.feats[begin:end])
        h_coords_list.append(h.coords[begin:end])
        h_feats_list.append(point_embeds.feats[i*10:(i+1)*10])
        h_coords_list.append(point_embeds.coords[i*10:(i+1)*10])
        begin = end + 10
    h = sp.SparseTensor(torch.cat(h_feats_list), torch.cat(h_coords_list))

    for block in self.blocks:
        h = block(h, t_emb, cond)

    h_feats_list = []
    h_coords_list = []
    begin = 0
    for i, coords_len in enumerate(coords_len_list):
        end = begin + 2 * coords_len
        h_feats_list.append(h.feats[begin:end])
        h_coords_list.append(h.coords[begin:end])
        begin = end
    h = sp.SparseTensor(torch.cat(h_feats_list), torch.cat(h_coords_list))

    h = manual_cast(h, x.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    # h.feats: [N, 1536]
    h = self.out_layer(h)
    # h.feats: [N, 32]
    return h

class Gen3DSeg(nn.Module):
    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model
        self.seg_embeddings = nn.Embedding(1, 1536)
        
    def get_positional_encoding(self, input_points):
        point_feats_embed = torch.zeros((10, 1536), dtype=torch.float32).to(input_points['point_slats'].feats.device)
        labels = input_points['point_labels'].squeeze(-1)
        point_feats_embed[labels == 1] = self.seg_embeddings.weight
        return sp.SparseTensor(point_feats_embed, input_points['point_slats'].coords)

    def forward(self, x_t, tex_slats, shape_slats, t, cond, input_points, coords_len_list):
        input_tex_feats_list = []
        input_tex_coords_list = []
        shape_feats_list = []
        shape_coords_list = []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            input_tex_feats_list.append(x_t.feats[begin:end])
            input_tex_feats_list.append(tex_slats.feats[begin:end])
            input_tex_coords_list.append(x_t.coords[begin:end])
            input_tex_coords_list.append(tex_slats.coords[begin:end])
            shape_feats_list.append(shape_slats.feats[begin:end])
            shape_feats_list.append(shape_slats.feats[begin:end])
            shape_coords_list.append(shape_slats.coords[begin:end])
            shape_coords_list.append(shape_slats.coords[begin:end])
            begin = end
        x_t = sp.SparseTensor(torch.cat(input_tex_feats_list), torch.cat(input_tex_coords_list))
        shape_slats = sp.SparseTensor(torch.cat(shape_feats_list), torch.cat(shape_coords_list))

        point_embeds = self.get_positional_encoding(input_points)
        output_tex_slats = self.flow_model(x_t, t, cond, shape_slats, point_embeds, coords_len_list)
        
        output_tex_feats_list = []
        output_tex_coords_list = []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            output_tex_feats_list.append(output_tex_slats.feats[begin:end])
            output_tex_coords_list.append(output_tex_slats.coords[begin:end])
            begin = begin + 2 * coords_len
        output_tex_slat = sp.SparseTensor(torch.cat(output_tex_feats_list), torch.cat(output_tex_coords_list))
        return output_tex_slat

def make_texture_square_pow2(img: Image.Image, target_size=None):
    w, h = img.size
    max_side = max(w, h)
    pow2 = 1
    while pow2 < max_side:
        pow2 *= 2
    if target_size is not None:
        pow2 = target_size
    pow2 = min(pow2, 2048)
    return img.resize((pow2, pow2), Image.BILINEAR)

def preprocess_scene_textures(asset):
    if not isinstance(asset, trimesh.Scene):
        return asset
    TEX_KEYS = ["baseColorTexture", "normalTexture", "metallicRoughnessTexture", "emissiveTexture", "occlusionTexture"]
    for geom in asset.geometry.values():
        visual = getattr(geom, "visual", None)
        mat = getattr(visual, "material", None)
        if mat is None:
            continue
        for key in TEX_KEYS:
            if not hasattr(mat, key):
                continue
            tex = getattr(mat, key)
            if tex is None:
                continue
            if isinstance(tex, Image.Image):
                setattr(mat, key, make_texture_square_pow2(tex))
            elif hasattr(tex, "image") and tex.image is not None:
                img = tex.image
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                tex.image = make_texture_square_pow2(img)
        if hasattr(mat, "image") and mat.image is not None:
            img = mat.image
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            mat.image = make_texture_square_pow2(img)
    return asset

def process_glb_to_vxz(glb_path, vxz_path):
    asset = trimesh.load(glb_path, force='scene')
    asset = preprocess_scene_textures(asset)
    aabb = asset.bounding_box.bounds
    center = (aabb[0] + aabb[1]) / 2
    scale = 0.99999 / (aabb[1] - aabb[0]).max()
    asset.apply_translation(-center)
    asset.apply_scale(scale)
    mesh = asset.to_mesh()
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices, faces, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False
    )
    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    voxel_indices = voxel_indices[mapping]
    dual_vertices = dual_vertices[mapping]
    intersected = intersected[mapping]

    voxel_indices_mat, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
        asset, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], timing=False
    )
    vid_mat = o_voxel.serialize.encode_seq(voxel_indices_mat)
    mapping_mat = torch.argsort(vid_mat)
    attributes = {k: v[mapping_mat] for k, v in attributes.items()}

    dual_vertices = dual_vertices * 512 - voxel_indices
    dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)

    attributes['dual_vertices'] = dual_vertices
    attributes['intersected'] = intersected
    o_voxel.io.write(vxz_path, voxel_indices, attributes)

def vxz_to_latent_slat(shape_encoder, shape_decoder, tex_encoder, vxz_path):
    coords, data = o_voxel.io.read(vxz_path)
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1).cuda()
    vertices = (data['dual_vertices'].cuda() / 255)
    intersected = torch.cat([data['intersected'] % 2, data['intersected'] // 2 % 2, data['intersected'] // 4 % 2], dim=-1).bool().cuda()
    vertices_sparse = sp.SparseTensor(vertices, coords)
    intersected_sparse = sp.SparseTensor(intersected.float(), coords)
    with torch.no_grad():
        shape_slat = shape_encoder(vertices_sparse, intersected_sparse)
        shape_slat = sp.SparseTensor(shape_slat.feats.cuda(), shape_slat.coords.cuda())
        shape_decoder.set_resolution(512)
        meshes, subs = shape_decoder(shape_slat, return_subs=True)
    
    base_color = (data['base_color'] / 255)
    metallic = (data['metallic'] / 255)
    roughness = (data['roughness'] / 255)
    alpha = (data['alpha'] / 255)
    attr = torch.cat([base_color, metallic, roughness, alpha], dim=-1).float().cuda() * 2 - 1
    with torch.no_grad():
        tex_slat = tex_encoder(sp.SparseTensor(attr, coords))
    return shape_slat, meshes, subs, tex_slat

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
    return {'cond': cond, 'neg_cond': neg_cond}

def tex_slat_sample_single(gen3dseg, sampler, pipeline_args, shape_slat, input_tex_slat, cond_dict, input_points):
    device = shape_slat.feats.device
    shape_std = torch.tensor(pipeline_args['shape_slat_normalization']['std'])[None].to(device)
    shape_mean = torch.tensor(pipeline_args['shape_slat_normalization']['mean'])[None].to(device)
    tex_std = torch.tensor(pipeline_args['tex_slat_normalization']['std'])[None].to(device)
    tex_mean = torch.tensor(pipeline_args['tex_slat_normalization']['mean'])[None].to(device)
    shape_slat = ((shape_slat - shape_mean) / shape_std)
    input_tex_slat = ((input_tex_slat - tex_mean) / tex_std)
    coords_len_list = [shape_slat.coords.shape[0]]
    noise = sp.SparseTensor(torch.randn_like(input_tex_slat.feats), shape_slat.coords)
    output_tex_slat = sampler.sample(gen3dseg, noise, input_tex_slat, shape_slat, input_points, coords_len_list, cond_dict, pipeline_args['tex_slat_sampler']['params'])
    output_tex_slat = output_tex_slat * tex_std + tex_mean
    return output_tex_slat

def slat_to_glb(meshes, tex_voxels, resolution=512):
    pbr_attr_layout = {
        'base_color': slice(0, 3),
        'metallic': slice(3, 4),
        'roughness': slice(4, 5),
        'alpha': slice(5, 6),
    }
    out_mesh = []
    for m, v in zip(meshes, tex_voxels):
        m.fill_holes()
        out_mesh.append(
            MeshWithVoxel(
                m.vertices, m.faces,
                origin = [-0.5, -0.5, -0.5],
                voxel_size = 1 / resolution,
                coords = v.coords[:, 1:],
                attrs = v.feats,
                voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                layout=pbr_attr_layout
            )
        )
    mesh = out_mesh[0]
    mesh.simplify(10000000)
    # mesh.simplify(16777216) # nvdiffrast limit
    glb = o_voxel.postprocess.to_glb(
        vertices            =   mesh.vertices,
        faces               =   mesh.faces,
        attr_volume         =   mesh.attrs,
        coords              =   mesh.coords,
        attr_layout         =   mesh.layout,
        voxel_size          =   mesh.voxel_size,
        aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target   =   100000, # 1000000
        texture_size        =   4096,
        remesh              =   True,
        # remesh              =   False,
        remesh_band         =   1,
        remesh_project      =   0,
        verbose             =   True
    )
    return glb

def inference(ckpt_path, item, input_vxz_points_list):
    print("-"*100)
    print("Loading model ............")
    with open("microsoft/TRELLIS.2-4B/pipeline.json", "r") as f:
        pipeline_config = json.load(f)
    pipeline_args = pipeline_config['args']
    tex_slat_flow_model = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    tex_slat_flow_model.forward = MethodType(flow_forward, tex_slat_flow_model)

    gen3dseg = Gen3DSeg(tex_slat_flow_model)
    state_dict = torch.load(ckpt_path)['state_dict']
    state_dict = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in state_dict.items()])
    gen3dseg.load_state_dict(state_dict)
    gen3dseg.eval()
    gen3dseg.cuda()
    sampler = Sampler()

    shape_encoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16").cuda().eval()
    tex_encoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16").cuda().eval()
    shape_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").cuda().eval()
    tex_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16").cuda().eval()

    rembg_model = BiRefNet(model_name="briaai/RMBG-2.0")
    rembg_model.cuda()
    image_cond_model = DinoV3FeatureExtractor(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
    image_cond_model.cuda()

    process_glb_to_vxz(item['glb'], item['input_vxz'])
    shape_slat, meshes, subs, tex_slat = vxz_to_latent_slat(shape_encoder, shape_decoder, tex_encoder, item['input_vxz'])

    print("-"*100)
    print("Getting cond ............")
    render_from_transforms(item['glb'], item['transforms'], item['img'])
    image = Image.open(item['img'])
    image = preprocess_image(rembg_model, image)
    cond = get_cond(image_cond_model, [image])

    print("-"*100)
    print("Sampling .................")
    vxz_points_coords = torch.tensor(input_vxz_points_list, dtype=torch.int32).cuda()
    vxz_points_coords = torch.cat([torch.zeros((vxz_points_coords.shape[0], 1), dtype=torch.int32).cuda(), vxz_points_coords], dim=1)
    input_points_coords = tex_encoder(sp.SparseTensor(torch.zeros((vxz_points_coords.shape[0], 6), dtype=torch.float32).cuda(), vxz_points_coords)).coords
    input_points_coords = torch.unique(input_points_coords, dim=0)
    point_num = input_points_coords.shape[0]
    if point_num >= 10:
        input_points_coords = input_points_coords[:10]
        point_labels = torch.tensor(([[1]]*10), dtype=torch.int32).cuda()
    else:
        input_points_coords = torch.cat([input_points_coords, torch.zeros((10 - point_num, 4), dtype=torch.int32).cuda()], dim=0)
        point_labels = torch.tensor(([[1]]*point_num+[[0]]*(10-point_num)), dtype=torch.int32).cuda()
    input_points = {'point_slats': sp.SparseTensor(input_points_coords, input_points_coords), 'point_labels': point_labels}

    output_tex_slat = tex_slat_sample_single(gen3dseg, sampler, pipeline_args, shape_slat, tex_slat, cond, input_points)
    with torch.no_grad():
        tex_voxels = tex_decoder(output_tex_slat, guide_subs=subs) * 0.5 + 0.5

    print("-"*100)
    print("Exporting glb ............")
    glb = slat_to_glb(meshes, tex_voxels)
    glb.export(item['export_glb'])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to trained checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--glb",
        type=str,
        required=True,
        help="Input glb path.",
    )
    parser.add_argument(
        "--input_vxz",
        type=str,
        required=True,
        help="Intermediate vxz path.",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        required=True,
        help="Path to transforms.json.",
    )
    parser.add_argument(
        "--img",
        type=str,
        required=True,
        help="Path to rendered input image.",
    )
    parser.add_argument(
        "--export_glb",
        type=str,
        required=True,
        help="Output glb path.",
    )
    parser.add_argument(
        "--input_vxz_points",
        type=int,
        nargs="+",
        required=True,
        help="List of voxel coordinates in vxz space in [0-511]^3 space, as a flat list of ints: x1 y1 z1 x2 y2 z2 ...",
    )

    args = parser.parse_args()
    if len(args.input_vxz_points) % 3 != 0:
        parser.error("--input_vxz_points length must be a multiple of 3 (x y z per point).")
    input_vxz_points_list = [
        args.input_vxz_points[i : i + 3]
        for i in range(0, len(args.input_vxz_points), 3)
    ]
    item = {
        "glb": args.glb,
        "input_vxz": args.input_vxz,
        "transforms": args.transforms,
        "img": args.img,
        "export_glb": args.export_glb,
    }
    inference(args.ckpt_path, item, input_vxz_points_list)
