import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import yaml

def plot_border_points_on_mask(border_points, image_array, output_directory, image_name):
    plot_image = np.stack([image_array] * 3, axis=-1)

    for point in border_points:
        x, y, sdf, location, shape_count = point
        if location == "inside":
            plot_image[x, y] = [255, 0, 0]
        else:
            plot_image[x, y] = [0, 0, 255]

    annotated_images_dir = os.path.join(output_directory, "annotated_images")
    os.makedirs(annotated_images_dir, exist_ok=True)

    annotated_image = Image.fromarray(plot_image.astype(np.uint8))

    annotated_image_path = os.path.join(annotated_images_dir, f"{image_name}_annotated.png")
    annotated_image.save(annotated_image_path)

    print(f"Image saved to: {annotated_image_path}")

def visualize_shape_with_latent(model, test_dataset, latent_code, shape_count, plots_dir, grid_size=224, grid_range=(-10, 10), device='cpu'):
    grid_size = int(grid_size)
    model.eval()
    with torch.no_grad():
        x = np.linspace(grid_range[0], grid_range[1], grid_size)
        y = np.linspace(grid_range[0], grid_range[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        points = torch.tensor(points, dtype=torch.float32).to(device)

        latent_code_expanded = latent_code.repeat(points.shape[0], 1)

        sdf_values = model.decoder(latent_code_expanded, points)
        sdf_values = sdf_values.cpu().numpy().reshape((grid_size, grid_size))

        sdf_values = np.rot90(sdf_values, k=1)

        image = np.zeros((grid_size, grid_size), dtype=np.uint8)
        image[sdf_values < 0] = 255

        shape_count = int(shape_count[0])
        annotated_image_path = f"shape{shape_count}_annotated.png"
        print(annotated_image_path)
        image_dir = os.path.join(test_dataset.root_folder, "data", 'annotated_images')
        image_path = os.path.join(image_dir, annotated_image_path)

        annotated_image = plt.imread(image_path)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(image, extent=(grid_range[0], grid_range[1], grid_range[0], grid_range[1]),
                       origin='lower', cmap='gray')
        axes[0].set_title('Reconstructed Shape')
        axes[0].axis('off')
        axes[0].set_aspect('equal')

        axes[1].imshow(annotated_image, cmap='gray')
        axes[1].set_title('Annotated Shape')
        axes[1].axis('off')
        axes[1].set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f"{plots_dir}/reconstructed_image_{shape_count}")
        plt.show()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)