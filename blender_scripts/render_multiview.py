import bpy
import random
import os
import mathutils
from mathutils import Vector
import math
import json

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

# Create the cameras and add them to the scene
for i in range(num_cameras):
    # Create a new camera object
    camera = bpy.data.cameras.new("Camera")
    
    # Set the FOV and aspect ratio of the camera
    camera.angle = camera_fov

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

    rot = direction.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    loc = mathutils.Matrix.Translation(camera_object.location)

    camera_object.matrix_world =  loc @ rot

    # Add the camera to the scene
    scene.collection.objects.link(camera_object)
    
    # Save the camera's world matrix and FOV to the dictionary
    camera_data.append({
        "transformation_matrix": [list(row) for row in camera_object.matrix_world],
        "fov": camera.angle,
        "file_path": "Camera_{}.png".format(camera_object.name)
    })

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
        scene.render.filepath = os.path.join(output_dir, "Camera_{}.png".format(camera_object.name))

        # Render the scene from the current camera
        bpy.ops.render.render(write_still=True)

# Remove the cameras from the scene
for camera_object in scene.objects:
    if camera_object.type == "CAMERA":
        scene.collection.objects.unlink(camera_object)
        
        # Delete the camera from memory
        bpy.data.cameras.remove(camera_object.data)