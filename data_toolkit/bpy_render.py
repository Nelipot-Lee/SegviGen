import bpy
import json
import math
import mathutils
import numpy as np

class BpyRenderer:
    def __init__(self, resolution=512, engine="CYCLES", geo_mode=False, split_normal=False):
        self.resolution = resolution
        self.engine = engine
        self.geo_mode = geo_mode
        self.split_normal = split_normal
        self.import_functions = self._setup_import_functions()

    def _setup_import_functions(self):
        import_functions = {
            "obj": bpy.ops.wm.obj_import,
            "glb": bpy.ops.import_scene.gltf,
            "gltf": bpy.ops.import_scene.gltf,
            "usd": bpy.ops.import_scene.usd,
            "fbx": bpy.ops.import_scene.fbx,
            "stl": bpy.ops.import_mesh.stl,
            "usda": bpy.ops.import_scene.usda,
            "dae": bpy.ops.wm.collada_import,
            "ply": bpy.ops.wm.ply_import,
            "abc": bpy.ops.wm.alembic_import,
            "blend": bpy.ops.wm.append,
        }
        return import_functions

    def init_render_settings(self):
        bpy.context.scene.render.engine = self.engine
        bpy.context.scene.render.resolution_x = self.resolution
        bpy.context.scene.render.resolution_y = self.resolution
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.film_transparent = True
        if self.engine == "CYCLES":
            bpy.context.scene.render.engine = "CYCLES"
            bpy.context.scene.cycles.samples = 128 if not self.geo_mode else 1
            bpy.context.scene.cycles.filter_type = "BOX"
            bpy.context.scene.cycles.filter_width = 1
            bpy.context.scene.cycles.diffuse_bounces = 1
            bpy.context.scene.cycles.glossy_bounces = 1
            bpy.context.scene.cycles.transparent_max_bounces = (3 if not self.geo_mode else 0)
            bpy.context.scene.cycles.transmission_bounces = (3 if not self.geo_mode else 1)
            bpy.context.scene.cycles.use_denoising = True
            try:
                bpy.context.scene.cycles.device = "GPU"
                bpy.context.preferences.addons["cycles"].preferences.get_devices()
                bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
            except:
                pass

    def init_scene(self):
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for material in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)
        for texture in bpy.data.textures:
            bpy.data.textures.remove(texture, do_unlink=True)
        for image in bpy.data.images:
            bpy.data.images.remove(image, do_unlink=True)

    def init_camera(self):
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        cam.data.sensor_height = cam.data.sensor_width = 32
        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        cam_empty = bpy.data.objects.new("Empty", None)
        cam_empty.location = (0, 0, 0)
        bpy.context.scene.collection.objects.link(cam_empty)
        cam_constraint.target = cam_empty
        return cam

    def init_lighting(self):
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
        bpy.context.collection.objects.link(default_light)
        default_light.data.energy = 1000
        default_light.location = (4, 1, 6)
        default_light.rotation_euler = (0, 0, 0)

        top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
        bpy.context.collection.objects.link(top_light)
        top_light.data.energy = 10000
        top_light.location = (0, 0, 10)
        top_light.scale = (100, 100, 100)

        bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
        bpy.context.collection.objects.link(bottom_light)
        bottom_light.data.energy = 1000
        bottom_light.location = (0, 0, -10)
        bottom_light.rotation_euler = (0, 0, 0)
        return {"default_light": default_light, "top_light": top_light, "bottom_light": bottom_light}

    def load_object(self, object_path):
        file_extension = object_path.split(".")[-1].lower()
        if file_extension not in self.import_functions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        import_function = self.import_functions[file_extension]
        print(f"Loading object from {object_path}")
        if file_extension == "blend":
            import_function(directory=object_path, link=False)
        elif file_extension in {"glb", "gltf"}:
            import_function(filepath=object_path, merge_vertices=True, import_shading="NORMALS")
        else:
            import_function(filepath=object_path)

    def delete_invisible_objects(self):
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.context.scene.objects:
            if obj.hide_viewport or obj.hide_render:
                obj.hide_viewport = False
                obj.hide_render = False
                obj.hide_select = False
                obj.select_set(True)
        bpy.ops.object.delete()
        invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
        for col in invisible_collections:
            bpy.data.collections.remove(col)

    def unhide_all_objects(self):
        for obj in bpy.context.scene.objects:
            obj.hide_set(False)

    def convert_to_meshes(self):
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
        for obj in bpy.context.scene.objects:
            obj.select_set(True)
        bpy.ops.object.convert(target="MESH")

    def triangulate_meshes(self):
        bpy.ops.object.select_all(action="DESELECT")
        objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        bpy.context.view_layer.objects.active = objs[0]
        for obj in objs:
            obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.reveal()
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

    def split_mesh_normal(self):
        bpy.ops.object.select_all(action="DESELECT")
        objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        bpy.context.view_layer.objects.active = objs[0]
        for obj in objs:
            obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.split_normals()
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

    def override_material(self):
        new_mat = bpy.data.materials.new(name="Override0123456789")
        new_mat.use_nodes = True
        new_mat.node_tree.nodes.clear()
        bsdf = new_mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
        bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
        bsdf.inputs[1].default_value = 1
        output = new_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        new_mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        bpy.context.scene.view_layers["View Layer"].material_override = new_mat

    def scene_bbox(self):
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False
        scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
        for obj in scene_meshes:
            found = True
            for coord in obj.bound_box:
                coord = mathutils.Vector(coord)
                coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        if not found:
            raise RuntimeError("no objects in scene to compute bounding box for")
        return mathutils.Vector(bbox_min), mathutils.Vector(bbox_max)

    def normalize_scene(self):
        scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
        if len(scene_root_objects) > 1:
            scene = bpy.data.objects.new("ParentEmpty", None)
            bpy.context.scene.collection.objects.link(scene)
            for obj in scene_root_objects:
                obj.parent = scene
        else:
            scene = scene_root_objects[0]

        bbox_min, bbox_max = self.scene_bbox()
        print(f"[INFO] Bounding box: {bbox_min}, {bbox_max}")
        scale = 1 / max(bbox_max - bbox_min)
        scene.scale = scene.scale * scale
        bpy.context.view_layer.update()
        bbox_min, bbox_max = self.scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        scene.matrix_world.translation += offset
        bpy.ops.object.select_all(action="DESELECT")
        return scale, offset

    def set_camera_from_matrix(self, cam, transform_matrix):
        matrix = mathutils.Matrix(transform_matrix)
        cam.matrix_world = matrix
        bpy.context.view_layer.update()

    def render_from_transforms(self, file_path, transforms_json_path, output_path):
        with open(transforms_json_path, 'r') as f:
            transforms_data = json.load(f)
        
        self.init_render_settings()
        
        if file_path.endswith(".blend"):
            self.delete_invisible_objects()
        else:
            self.init_scene()
            self.load_object(file_path)
            if self.split_normal:
                self.split_mesh_normal()
        print("[INFO] Scene initialized.")
        
        scale, offset = self.normalize_scene()
        print(f"[INFO] Scene normalized with auto scale: {scale}, offset: {offset}")
        
        cam = self.init_camera()
        self.init_lighting()
        print("[INFO] Camera and lighting initialized.")
        if self.geo_mode:
            self.override_material()

        transform_matrix = transforms_data[0]["transform_matrix"]
        camera_angle_x = transforms_data[0]["camera_angle_x"]
        self.set_camera_from_matrix(cam, transform_matrix)
        if camera_angle_x is not None:
            cam.data.lens = 16 / np.tan(camera_angle_x / 2)
        
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

def render_from_transforms(file_path, transforms_json_path, output_path, resolution=512, engine="CYCLES", geo_mode=False, split_normal=False):
    renderer = BpyRenderer(resolution=resolution, engine=engine, geo_mode=geo_mode, split_normal=split_normal)
    return renderer.render_from_transforms(file_path, transforms_json_path, output_path)

if __name__ == "__main__":
    file_path = "./assets/example.glb"
    transforms_json_path = "transforms.json"
    output_path = "./assets/img.png"
    render_from_transforms(file_path=file_path, transforms_json_path=transforms_json_path, output_path=output_path)