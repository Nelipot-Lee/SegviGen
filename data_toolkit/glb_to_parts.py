import os
import trimesh

def glb_to_parts(glb_path, output_dir):
    scene = trimesh.load(glb_path, force='scene')
    os.makedirs(output_dir, exist_ok=True)
    geometries = list(scene.geometry.values())
    for idx, geometry in enumerate(geometries):
        part_scene = trimesh.Scene()
        part_scene.add_geometry(geometry)
        output_path = os.path.join(output_dir, f"{idx}.glb")
        part_scene.export(output_path)

if __name__ == "__main__":
    glb_path = "./assets/example.glb"
    output_dir = "./assets/parts"
    glb_to_parts(glb_path, output_dir)