import os
import json
import random
import trimesh
import numpy as np

from trimesh.visual.material import PBRMaterial

def _load_as_single_mesh(part_path):
    obj = trimesh.load(part_path, force="scene")
    if isinstance(obj, trimesh.Scene):
        dumped = obj.dump()
        meshes = [m for m in dumped if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
        return trimesh.util.concatenate(meshes)
    if isinstance(obj, trimesh.Trimesh):
        return obj

def set_mesh_solid_pbr(mesh, rgba_uint8=(255, 255, 255, 255), emissive=True):
    rgb = np.array(rgba_uint8[:3], dtype=np.float32) / 255.0
    a = float(rgba_uint8[3]) / 255.0
    colors = np.tile(np.array(rgba_uint8, dtype=np.uint8), (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    mat_kwargs = dict(
        baseColorFactor=[float(rgb[0]), float(rgb[1]), float(rgb[2]), a],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    if emissive:
        mat_kwargs["emissiveFactor"] = [float(rgb[0]), float(rgb[1]), float(rgb[2])]
    mesh.visual.material = PBRMaterial(**mat_kwargs)
    return mesh

def color_glb(parts_path, output_path, interactive):
    part_meshes = []
    for part_name in sorted(os.listdir(parts_path)):
        part_path = os.path.join(parts_path, part_name)
        part_meshes.append(_load_as_single_mesh(part_path))

    if interactive:
        for i in range(len(part_meshes)):
            colors = [(255, 255, 255, 255) if j == i else (0, 0, 0, 255) for j in range(len(part_meshes))]
            scene = trimesh.Scene()
            for j, m in enumerate(part_meshes):
                mc = m.copy()
                set_mesh_solid_pbr(mc, rgba_uint8=colors[j], emissive=True)
                scene.add_geometry(mc, node_name=f"part_{j}", geom_name=f"geom_{j}")
                os.makedirs(os.path.join(output_path, f"{i}"), exist_ok=True)
                scene.export(os.path.join(output_path, f"{i}", "output.glb"))
    else:
        colors = []
        for i in range(len(part_meshes)):
            while True:
                rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
                if rgb not in colors:
                    colors.append(rgb)
                    break

        # colors_base = [           # Static colors
        #     (0, 0, 0, 255),
        #     (0, 0, 255, 255),
        #     (0, 255, 0, 255),
        #     (0, 255, 255, 255),
        #     (255, 0, 0, 255),
        #     (255, 0, 255, 255),
        #     (255, 255, 0, 255),
        #     (255, 255, 255, 255)
        # ]
        # colors = random.sample(colors_base, len(part_meshes))
        
        with open(os.path.join(output_path, "colors.json"), "w", encoding="utf-8") as f:
            json.dump([list(c) for c in colors], f, ensure_ascii=False, indent=4)

        scene = trimesh.Scene()
        for i, m in enumerate(part_meshes):
            mc = m.copy()
            set_mesh_solid_pbr(mc, rgba_uint8=colors[i], emissive=True)
            scene.add_geometry(mc, node_name=f"part_{i}", geom_name=f"geom_{i}")
        scene.export(os.path.join(output_path, "output.glb"))

if __name__ == "__main__":
    parts_path = "./assets/parts"
    output_path = "./assets/interactive_seg"
    interactive = True
    color_glb(parts_path, output_path, interactive)