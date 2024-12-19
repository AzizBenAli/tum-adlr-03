import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from matplotlib.animation import FuncAnimation
from scripts.data_manipulation.data_loader import DeepSDFDataset2D

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

def interpolate_latent_codes(z1, z2, num_steps=10):
    interpolated_latents = []
    for t in np.linspace(0, 1, num_steps):
        z_t = (1 - t) * z1 + t * z2
        interpolated_latents.append(z_t)
    return interpolated_latents

def slerp(z1, z2, num_steps=10):
    z1_norm = z1 / torch.norm(z1)
    z2_norm = z2 / torch.norm(z2)
    omega = torch.acos(torch.clamp(torch.dot(z1_norm.squeeze(), z2_norm.squeeze()), -1.0 + 1e-7, 1.0 - 1e-7))
    sin_omega = torch.sin(omega)
    interpolated_latents = []
    for t in np.linspace(0, 1, num_steps):
        if sin_omega == 0:
            z_t = (1 - t) * z1 + t * z2
        else:
            z_t = (torch.sin((1 - t) * omega) / sin_omega) * z1 + (torch.sin(t * omega) / sin_omega) * z2
        interpolated_latents.append(z_t)
    return interpolated_latents

def generate_shape_image(model, latent_code, grid_size=224, grid_range=(-10, 10), device='cpu'):
    grid_size = int(grid_size)
    model.eval()
    with torch.no_grad():
        x = np.linspace(grid_range[0], grid_range[1], grid_size)
        y = np.linspace(grid_range[0], grid_range[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([yy.ravel(),xx.ravel()], axis=-1)
        points = torch.tensor(points, dtype=torch.float32).to(device)

        latent_code_expanded = latent_code.repeat(points.shape[0], 1)

        sdf_values = model.decoder(latent_code_expanded, points)
        sdf_values = sdf_values.cpu().numpy().reshape((grid_size, grid_size))

        image = np.zeros((grid_size, grid_size), dtype=np.uint8)
        image[sdf_values < 0] = 255

    return image

def create_interpolation_animation(model, interpolated_latents, grid_size=224, grid_range=(-10, 10),
                                   device='cpu', save_path=None):
    images = []
    for z in interpolated_latents:
        image = generate_shape_image(model, z, grid_size, grid_range, device)
        images.append(image)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    img_display = ax.imshow(images[0], cmap='gray', animated=True)

    def update(frame):
        img_display.set_array(images[frame])
        return [img_display]

    anim = FuncAnimation(fig, update, frames=len(images), interval=200, blit=True)

    if save_path:
        anim.save(save_path, writer='pillow', fps=5)

    plt.close(fig)
    return anim

def visualize_latent_space(model, device='cpu'):
    latent_codes = model.latent_codes.weight.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    latent_codes_2d = tsne.fit_transform(latent_codes)

    plt.figure(figsize=(10, 8))
    plt.scatter(latent_codes_2d[:, 0], latent_codes_2d[:, 1], alpha=0.7)
    plt.title('Latent Space Visualization using t-SNE')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

def search_train_dataset(shape_idx):
    train_dataset = DeepSDFDataset2D('../../multi_class/data', split='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    for batch in train_loader:
        batch_shape_idx = batch['shape_idx']
        if batch_shape_idx[0].item() == shape_idx:
            return batch['shape_count']
    else:
        print(f"Shape index {shape_idx} not found in test data.")
        return