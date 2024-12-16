import os
import cv2
import csv
import trimesh
import io
import random
from utils import *

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Global tracker for shape count
shape_tracker = 1

class MeshProcessor:
    def __init__(self, parent_directory, output_directory, max_objects=100, resolution=(224, 224), rotation_angle=20, border_threshold=10.0):
        self.parent_directory = parent_directory
        self.output_directory = output_directory
        self.max_objects = max_objects  # Limit on number of objects to process
        self.resolution = resolution
        self.rotation_angle = np.radians(rotation_angle)
        self.border_threshold = border_threshold

    def process_mesh(self, mesh_path, shape_tracker):
        # Load the mesh
        mesh = trimesh.load_mesh(mesh_path)

        # Create the scene and add geometry
        scene = trimesh.Scene()
        scene.add_geometry(mesh)

        # Set up camera parameters
        points = np.array([[-2.5, 0, 5]])
        fov = np.array([60.0, 60.0])
        camera = trimesh.scene.cameras.Camera(resolution=self.resolution, fov=fov)
        camera_transform = camera.look_at(points)

        # Apply transformations
        zoom_factor = -6
        zoom_matrix = trimesh.transformations.translation_matrix([0, 0, zoom_factor])
        camera_transform = np.dot(camera_transform, zoom_matrix)

        side_camera_position = np.array([3.0, 0.0, 0.0])
        translation_matrix = trimesh.transformations.translation_matrix(side_camera_position)
        camera_transform = np.dot(camera_transform, translation_matrix)

        angle = 150
        rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0])  # Y-axis rotation
        camera_transform = np.dot(camera_transform, rotation_matrix)

        # Update the scene camera
        scene.camera_transform = camera_transform

        # Render and process the image
        image_bytes = scene.save_image(resolution=self.resolution)
        image = Image.open(io.BytesIO(image_bytes))
        image_gray = image.convert("L")
        image_array = np.array(image_gray)

        # Generate binary mask and signed distance field (SDF)
        _, binary_mask = cv2.threshold(image_array, 254, 255, cv2.THRESH_BINARY)
        dist_inside = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(255 - binary_mask, cv2.DIST_L2, 5)
        sdf = dist_outside - dist_inside

        # Collect border and random points
        border_points = [
            [i, j, sdf[i, j], "inside" if sdf[i, j] < 0 else "outside", shape_tracker]
            for i in range(sdf.shape[0])
            for j in range(sdf.shape[1])
            if abs(sdf[i, j]) < self.border_threshold
        ]

        num_border_points = 2000
        border_points_sampled = random.sample(border_points, min(len(border_points), num_border_points))

        num_random_points = 1000
        filtered_points = [
            (i, j) for i in range(sdf.shape[0]) for j in range(sdf.shape[1]) if abs(sdf[i, j]) < 3
        ]
        random_points = random.sample(filtered_points, min(len(filtered_points), num_random_points))

        final_points = border_points_sampled + [
            [i, j, sdf[i, j], "inside" if sdf[i, j] < 0 else "outside", shape_tracker]
            for i, j in random_points
        ]

        # Save CSV file
        csv_dir = os.path.join(self.output_directory, "csv_files")
        os.makedirs(csv_dir, exist_ok=True)
        csv_filename = os.path.join(csv_dir, f"shape{shape_tracker}.csv")
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y", "sdf", "location", "shape_count"])
            writer.writerows(final_points)

        # Save the image
        image_dir = os.path.join(self.output_directory, "images")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = os.path.join(image_dir, f"shape{shape_tracker}.png")
        image.save(image_filename)

        # Plot and save border points visualization
        plot_border_points_on_mask(border_points_sampled, image_array, self.output_directory, f"shape{shape_tracker}")

    def process_folder(self, shape_name):
        global shape_tracker
        folder_path = os.path.join(self.parent_directory, shape_name)
        processed_count = 0  # Count of processed objects for this shape type

        for root, dirs, files in os.walk(folder_path):
            if "models" in dirs:
                models_folder = os.path.join(root, "models")
                for file_name in os.listdir(models_folder):
                    if file_name.endswith(".obj"):
                        if processed_count >= self.max_objects:
                            print(f"Reached max_objects ({self.max_objects}) for {shape_name}.")
                            return

                        obj_file_path = os.path.join(models_folder, file_name)
                        print(f"Processing {obj_file_path}")
                        self.process_mesh(obj_file_path, shape_tracker)
                        shape_tracker += 1  # Increment the global tracker
                        processed_count += 1


if __name__ == "__main__":
    parent_directory = "/Users/yahyaabdelhamed"
    shapes = ["cars", "chairs", "lamps"]
    output_base_dir = "../data_shapeNet/data"
    os.makedirs(output_base_dir, exist_ok=True)

    # Define the maximum number of objects to process per type
    max_objects_per_type = 300

    processor = MeshProcessor(parent_directory=parent_directory, output_directory=output_base_dir, max_objects=max_objects_per_type)

    for shape in shapes:
        processor.process_folder(shape)
