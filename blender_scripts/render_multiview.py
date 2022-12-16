import bpy
import random
import os
import mathutils
from mathutils import Vector
import math
import json

import cv2
import numpy as np

# Set the output directory for the rendered images
output_dir = "/Users/peterstefek/workspace/NeRF/data/cube"

json_file = os.path.join(output_dir, "cameras.json")

# Get the current scene
scene = bpy.context.scene

# Set the number of cameras to create
num_cameras = 50

# Set the center position of the cameras
center_position = Vector((0, 0, 0))

# Set the range of possible camera positions
r = 5

camera_data = []

camera_fov = math.radians(45)

render_width = 200
render_height = 200

# Setup render tree
# switch on nodes
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# create input render layer node
rl = tree.nodes.new("CompositorNodeRLayers")

# create output node
v = tree.nodes.new("CompositorNodeViewer")
v.use_alpha = False

min_depth = 0.5
max_depth = 7

# Links
links.new(rl.outputs[2], v.inputs[0])  # link Depth output to Viewer input

# Create the cameras and add them to the scene
for i in range(num_cameras):
    # Create a new camera object
    camera = bpy.data.cameras.new("Camera")

    # Set the FOV and aspect ratio of the camera
    camera.angle = camera_fov

    camera.clip_start = 0.1
    camera.clip_end = 10

    # Create a new object for the camera
    camera_object = bpy.data.objects.new("Camera", camera)

    # Set the camera's position on the surface of the sphere
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, math.pi)
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    camera_object.location = (x, y, z)

    # Point the camera at the center position
    direction = camera_object.location - center_position

    rot = direction.to_track_quat("Z", "Y").to_matrix().to_4x4()
    loc = mathutils.Matrix.Translation(camera_object.location)

    camera_object.matrix_world = loc @ rot

    # Add the camera to the scene
    scene.collection.objects.link(camera_object)

    # Save the camera's world matrix and FOV to the dictionary
    camera_data.append(
        {
            "transformation_matrix": [list(row) for row in camera_object.matrix_world],
            "fov": camera.angle,
            "file_path": "Camera_{}.png".format(camera_object.name),
        }
    )

# Set the active camera in the scene
scene.camera = camera_object

# Set the render size
scene.render.resolution_x = render_width
scene.render.resolution_y = render_height

# Save the camera data to the JSON file
with open(json_file, "w") as f:
    json.dump({"frames": camera_data}, f)

# Render the scene from each of the cameras
for camera_object in scene.objects:
    if camera_object.type == "CAMERA":
        # Set the current camera as the active camera
        scene.camera = camera_object

        # Set the output path for the current camera
        scene.render.filepath = os.path.join(
            output_dir, "Camera_{}.png".format(camera_object.name)
        )

        # Render the scene from the current camera
        bpy.ops.render.render(write_still=True)

        pixels = bpy.data.images["Viewer Node"].pixels

        # copy buffer to numpy array for faster manipulation
        arr = np.array(pixels[:])
        arr = arr.reshape((render_width, render_height, -1))[:, :, 0]
        print(arr, np.min(arr), np.max(arr))
        arr[arr > max_depth] = max_depth
        arr[arr < min_depth] = min_depth
        arr -= min_depth
        arr /= max_depth - min_depth
        arr = (1.0 - arr) * 255
        arr = np.flip(arr, axis=0)
        print(np.max(arr), np.min(arr))
        assert np.max(arr) <= 255
        assert np.min(arr) >= 0
        print(arr.shape)

        cv2.imwrite(
            os.path.join(output_dir, "Camera_{}_depth.png".format(camera_object.name)),
            arr,
        )
        print(
            "saved:",
            os.path.join(output_dir, "Camera_{}_depth.png".format(camera_object.name)),
        )


# Remove the cameras from the scene
for camera_object in scene.objects:
    if camera_object.type == "CAMERA":
        scene.collection.objects.unlink(camera_object)

        # Delete the camera from memory
        bpy.data.cameras.remove(camera_object.data)
