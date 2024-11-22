import trimesh
import numpy as np
from PIL import Image
import io

mesh = trimesh.load_mesh('/Users/vw4eg83/cars/1a30678509a1fdfa7fb5267a071ae23a/models/model_normalized.obj')

# Create a scene
scene = trimesh.Scene()

# Add the loaded mesh to the scene
scene.add_geometry(mesh)

points = np.array([[-2.5,0, 5]])  # Or use mesh.centroid to center the camera on the mesh

# Set the camera's field of view (fov in degrees)
fov = np.array([60.0, 60.0])  # Example FOV in x and y

# Create a camera
camera = trimesh.scene.cameras.Camera(resolution=(224, 224), fov=fov)

# Create the camera transform to look at the points
camera_transform = camera.look_at(points)

# Apply a zoom-out effect by translating the camera along the Z-axis (away from the object)
zoom_factor =-6 # Zoom out factor (larger value = more zoom out)
zoom_matrix = trimesh.transformations.translation_matrix([0, 0, zoom_factor])

# Apply the zoom-out transformation
camera_transform = np.dot(camera_transform, zoom_matrix)

side_camera_position = np.array([3.0, 0.0, 0.0])  # Adjust the position on the X-axis to view the side


translation_matrix = trimesh.transformations.translation_matrix(side_camera_position)
camera_transform = np.dot(camera_transform, translation_matrix)

angle = 150
rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0])  # Y-axis rotation

camera_transform = np.dot(camera_transform, rotation_matrix)

scene.camera_transform = camera_transform

image_bytes = scene.save_image(resolution=[224, 224])

image_pil = Image.open(io.BytesIO(image_bytes))

image_pil.show()
# S
