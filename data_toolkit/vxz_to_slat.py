import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import o_voxel
import trellis2.modules.sparse as sp

from trellis2 import models

def vxz_to_latent_slat(shape_encoder, tex_encoder, vxz_path, return_foreground=False):
    coords, data = o_voxel.io.read(vxz_path)
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1).cuda()
    vertices = (data['dual_vertices'].cuda() / 255)
    intersected = torch.cat([data['intersected'] % 2, data['intersected'] // 2 % 2, data['intersected'] // 4 % 2], dim=-1).bool().cuda()
    vertices_sparse = sp.SparseTensor(vertices, coords)
    intersected_sparse = sp.SparseTensor(intersected.float(), coords)
    with torch.no_grad():
        shape_slat = shape_encoder(vertices_sparse, intersected_sparse)
        shape_slat = sp.SparseTensor(shape_slat.feats.cuda(), shape_slat.coords.cuda())
    
    base_color = (data['base_color'] / 255)
    metallic = (data['metallic'] / 255)
    roughness = (data['roughness'] / 255)
    alpha = (data['alpha'] / 255)
    attr = torch.cat([base_color, metallic, roughness, alpha], dim=-1).float().cuda() * 2 - 1
    with torch.no_grad():
        tex_slat = tex_encoder(sp.SparseTensor(attr, coords))
        if return_foreground:
            mask = ((base_color == 1.0).sum(dim=1) == 3)
            neg_mask = ((base_color != 1.0).sum(dim=1) == 3)
            tex_slat_foreground = tex_encoder(sp.SparseTensor(attr[mask], coords[mask]))
            tex_slat_background = tex_encoder(sp.SparseTensor(attr[neg_mask], coords[neg_mask]))
            foreground_coords = torch.unique(tex_slat_foreground.coords, dim=0)
            background_coords = torch.unique(tex_slat_background.coords, dim=0)
            N = background_coords.shape[0]
            all_coords = torch.cat([background_coords, foreground_coords], dim=0)
            _, inv = torch.unique(all_coords, dim=0, return_inverse=True)
            inv_background = inv[:N]
            inv_foreground = inv[N:]
            keep = ~torch.isin(inv_foreground, inv_background)
            foreground_coords = foreground_coords[keep]
    if return_foreground:
        return shape_slat, tex_slat, foreground_coords
    else:
        return shape_slat, tex_slat

def get_common_coords(slat1, slat2, slat3, slat4, foreground_coords_origin=None):
    coords_list = [slat1.coords, slat2.coords, slat3.coords, slat4.coords]
    xs = [torch.unique(x, dim=0) for x in coords_list]
    all_coords = torch.cat(xs, dim=0)
    uniq_coords, counts = torch.unique(all_coords, dim=0, return_counts=True)
    common_coords = uniq_coords[counts == len(coords_list)].cuda()

    if foreground_coords_origin is not None:
        xs_foreground = [torch.unique(x, dim=0) for x in (common_coords, foreground_coords_origin)]
        all_coords_foreground = torch.cat(xs_foreground, dim=0)
        uniq_coords_foreground, counts_foreground = torch.unique(all_coords_foreground, dim=0, return_counts=True)
        foreground_coords = uniq_coords_foreground[counts_foreground == 2].cuda()
        return common_coords, foreground_coords
    else:
        return common_coords

def get_slat_by_common_coords(slat_origin, common_coords):
    N = slat_origin.coords.shape[0]
    all_coords = torch.cat([slat_origin.coords, common_coords], dim=0)
    uniq_coords, inv = torch.unique(all_coords, dim=0, return_inverse=True)
    inv_slat = inv[:N].cuda()
    inv_common = inv[N:].cuda()
    device = slat_origin.coords.device
    idx_map = torch.full((uniq_coords.shape[0],), -1, dtype=torch.int32, device=device)
    slat_idx = torch.arange(N, dtype=torch.int32, device=device)
    idx_map.scatter_reduce_(0, inv_slat, slat_idx, reduce='amin', include_self=False)
    idx_in_slat = idx_map[inv_common]
    feats = slat_origin.feats[idx_in_slat]
    return sp.SparseTensor(feats, common_coords)

def get_point(point_num, common_coords, foreground_coords):
    device = common_coords.device
    point_feats_coords = torch.zeros((10, 4), dtype=torch.int32, device=device)
    point_labels = torch.zeros((10, 1), dtype=torch.int32, device=device)
    foreground_idx = torch.randperm(foreground_coords.shape[0], device=device)[:point_num]
    point_foreground = foreground_coords[foreground_idx]
    if point_foreground.shape[0] != point_num:
        return None
    point_feats_coords[:point_num] = point_foreground
    point_labels[:point_num] = 1
    return {'point_feats': point_feats_coords.cpu(), 'point_labels': point_labels.cpu()}

def vxz_to_slat(shape_encoder, tex_encoder, input_vxz_path, output_vxz_path, save_dir, interactive):
    input_shape_slat_origin, input_tex_slat_origin = vxz_to_latent_slat(shape_encoder, tex_encoder, input_vxz_path)
    if interactive:
        output_shape_slat_origin, output_tex_slat_origin, foreground_coords_origin = vxz_to_latent_slat(shape_encoder, tex_encoder, output_vxz_path, return_foreground=interactive)
        common_coords, foreground_coords = get_common_coords(input_shape_slat_origin, input_tex_slat_origin, output_shape_slat_origin, output_tex_slat_origin, foreground_coords_origin)
    else:
        output_shape_slat_origin, output_tex_slat_origin = vxz_to_latent_slat(shape_encoder, tex_encoder, output_vxz_path)
        common_coords = get_common_coords(input_shape_slat_origin, input_tex_slat_origin, output_shape_slat_origin, output_tex_slat_origin)

    os.makedirs(save_dir, exist_ok=True)
    shape_slat = get_slat_by_common_coords(input_shape_slat_origin, common_coords)
    torch.save({"feats": shape_slat.feats.cpu(), "coords": shape_slat.coords.cpu()}, os.path.join(save_dir, "shape_slat.pth"))
    input_tex_slat = get_slat_by_common_coords(input_tex_slat_origin, common_coords)
    torch.save({"feats": input_tex_slat.feats.cpu(), "coords": input_tex_slat.coords.cpu()}, os.path.join(save_dir, "input_tex_slat.pth"))
    output_tex_slat_gt = get_slat_by_common_coords(output_tex_slat_origin, common_coords)
    torch.save({"feats": output_tex_slat_gt.feats.cpu(), "coords": output_tex_slat_gt.coords.cpu()}, os.path.join(save_dir, "output_tex_slat.pth"))
    
    if interactive:
        for point_num in range(1, 11):
            input_points = get_point(point_num, common_coords, foreground_coords)
            if input_points is None:
                continue
            torch.save(input_points, os.path.join(save_dir, f"point_{point_num}.pth"))

if __name__ == "__main__":
    input_vxz_path = "./assets/input.vxz"
    output_vxz_path = "./assets/interactive_seg/0/output.vxz"
    save_dir = "./assets/interactive_seg/0"
    interactive = True
    shape_encoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16").cuda().eval()
    tex_encoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16").cuda().eval()
    vxz_to_slat(shape_encoder, tex_encoder, input_vxz_path, output_vxz_path, save_dir, interactive)