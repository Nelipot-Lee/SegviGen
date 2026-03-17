import os
import json

import numpy as np
import trimesh
import torch
import nvdiffrast.torch as nr
from PIL import Image

def build_projection_matrix(fov, width, height, z_near=0.01, z_far=100.0):
    aspect = float(width) / float(height)
    fov_y = 2.0 * np.arctan(np.tan(fov / 2.0) / aspect)
    f = 1.0 / np.tan(fov_y / 2.0)
    P = np.array([
        [f / aspect, 0.0,  0.0,                                 0.0],
        [0.0,        f,    0.0,                                 0.0],
        [0.0,        0.0,  (z_far + z_near) / (z_near - z_far), (2.0 * z_far * z_near) / (z_near - z_far)],
        [0.0,        0.0, -1.0,                                 0.0],
    ], dtype=np.float32)
    return P

def compute_bbox_center_and_scale_like_blender(vertices):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_extents = bbox_max - bbox_min
    scale = 1.0 / np.max(bbox_extents)
    offset = -(bbox_min + bbox_max) / 2.0
    return offset, scale

def _load_as_single_mesh(part_path):
    obj = trimesh.load(part_path, force="scene")
    if isinstance(obj, trimesh.Scene):
        dumped = obj.dump()
        meshes = [m for m in dumped if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
        return trimesh.util.concatenate(meshes)
    if isinstance(obj, trimesh.Trimesh):
        return obj

def load_parts_from_directory(object_path):
    per_part_vertices = []
    per_part_faces = []
    part_names = []
    vertices_counts = []
    vertex_offset = 0
    for part_name in sorted(os.listdir(object_path)):
        part_path = os.path.join(object_path, part_name)
        mesh = _load_as_single_mesh(part_path)
        v = mesh.vertices.astype(np.float32)
        f = mesh.faces.astype(np.int32)
        per_part_vertices.append(v)
        per_part_faces.append(f + vertex_offset)
        part_names.append(part_name)
        vertices_counts.append(v.shape[0])
        vertex_offset += v.shape[0]
    return per_part_vertices, per_part_faces, part_names, vertices_counts

def save_png(path, array_uint8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array_uint8, mode="RGB").save(path)

def render_views(glctx, V, F, C, output_path, transforms_path):
    width = 512
    height = 512
    fov = 40.0*np.pi/180.0
    V_t = torch.from_numpy(V).to(torch.float32).cuda()
    F_t = torch.from_numpy(F).to(torch.int32).cuda()
    C_t = torch.from_numpy(C).to(torch.float32).cuda()
    theta = np.pi / 2.0
    Gx = np.array([
        [1.0, 0.0,             0.0,            0.0],
        [0.0, np.cos(theta),  -np.sin(theta),  0.0],
        [0.0, np.sin(theta),   np.cos(theta),  0.0],
        [0.0, 0.0,             0.0,            1.0],
    ], dtype=np.float32)
    Gx_t = torch.from_numpy(Gx).to(torch.float32).cuda()

    with open(transforms_path, "r") as f:
        transforms = json.load(f)
    cam_to_world = np.array(transforms[0]["transform_matrix"], dtype=np.float32)
    world_to_cam = np.linalg.inv(cam_to_world)
    P = build_projection_matrix(fov, width, height)

    V_mat = torch.from_numpy(world_to_cam).to(torch.float32).cuda()
    P_mat = torch.from_numpy(P).to(torch.float32).cuda()
    M_t = torch.eye(4, dtype=torch.float32).cuda()
    pos_h = torch.cat([V_t, torch.ones((V_t.shape[0], 1), dtype=torch.float32).cuda()], dim=1)
    pos_clip = (P_mat @ V_mat @ M_t @ Gx_t) @ pos_h.t()
    pos_clip = pos_clip.t().contiguous().unsqueeze(0)

    rast, _ = nr.rasterize(glctx, pos_clip, F_t, resolution=[height, width])
    feat, _ = nr.interpolate(C_t.unsqueeze(0), rast, F_t)
    cov = rast[..., 3:4]
    img = feat.clamp(0.0, 1.0)
    bg = torch.ones_like(img)
    out = img * (cov > 0) + bg * (cov <= 0)
    out_np = (out[0].cpu().numpy() * 255.0).astype(np.uint8)
    out_np = out_np[::-1, :, :]
    save_png(output_path, out_np)

def color_img(object_path, output_path, transforms, colors_path):
    per_part_vertices, per_part_faces, part_names, vertices_counts = load_parts_from_directory(object_path)
    V = np.concatenate(per_part_vertices, axis=0).astype(np.float32)
    F = np.concatenate(per_part_faces, axis=0).astype(np.int32)
    offset, scale = compute_bbox_center_and_scale_like_blender(V)
    V_scaled = V * scale
    V_norm = V_scaled + offset[None, :]
    V = V_norm

    with open(colors_path, "r") as f:
        external_colors = json.load(f)
    color_map = {}
    colors = []
    for idx, part_name in enumerate(part_names):
        rgb = external_colors[idx][:3]
        color_map[part_name] = [int(rgb[0]), int(rgb[1]), int(rgb[2])]
        num_v = vertices_counts[idx]
        col = (np.array(rgb, dtype=np.float32) / 255.0)[None, :]
        colors.append(np.repeat(col, repeats=num_v, axis=0))

    C = np.concatenate(colors, axis=0).astype(np.float32)
    glctx = nr.RasterizeCudaContext()
    render_views(glctx, V, F, C, output_path, transforms)

if __name__ == "__main__":
    object_path = "./assets/parts"
    output_path = "./assets/full_seg_w_2d_map/2d_map.png"
    transforms = "transforms.json"
    colors_path = "./assets/full_seg_w_2d_map/colors.json"
    color_img(object_path, output_path, transforms, colors_path)