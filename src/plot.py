import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import DeepSDFModel  # Ensure this path is correct
from torch.utils.data import DataLoader
from src.output.Data_loader import DeepSDFDataset2D  # Adjust import based on your project structure
import random
import cv2


# -----------------------------
# Utility Functions
# -----------------------------

def generate_grid(grid_size=128, range_min=-1.0, range_max=1.0):
    """
    Generate a 2D grid of points covering the image space.

    Args:
        grid_size (int): Resolution of the grid.
        range_min (float): Minimum coordinate value.
        range_max (float): Maximum coordinate value.

    Returns:
        grid_points (torch.Tensor): Tensor of shape (grid_size**2, 2).
        grid_x, grid_y (numpy.ndarray): Meshgrid for plotting.
    """
    x = np.linspace(range_min, range_max, grid_size)
    y = np.linspace(range_min, range_max, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    return torch.tensor(grid_points, dtype=torch.float32), grid_x, grid_y


def plot_reconstructed_borders(sdf_grid, grid_x, grid_y, title, save_path=None):
    """
    Plot the reconstructed border for a given SDF grid.

    Args:
        sdf_grid (numpy.ndarray): SDF grid of shape (grid_size, grid_size).
        grid_x, grid_y (numpy.ndarray): Meshgrid for plotting.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot image.
    """
    plt.figure(figsize=(6, 6))

    # Create a binary image based on SDF values
    # Negative SDF (inside) as white, Positive SDF (outside) as black
    binary_image = np.where(sdf_grid < 0, 1, 0)  # 1 for inside (white), 0 for outside (black)

    plt.imshow(binary_image, extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
               origin='lower', cmap='gray', alpha=0.5)

    # Overlay the zero-level contour
    plt.contour(grid_x, grid_y, sdf_grid, levels=[0], colors='red')

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path)
    plt.show()


def load_latent_codes(latent_codes_path):
    """
    Load latent codes from a numpy file.

    Args:
        latent_codes_path (str): Path to the latent_codes.npy file.

    Returns:
        latent_codes (numpy.ndarray): Array of latent codes.
    """
    if not os.path.exists(latent_codes_path):
        raise FileNotFoundError(f"Latent codes file not found at {latent_codes_path}.")
    latent_codes = np.load(latent_codes_path)
    return latent_codes


def load_image_paths(output_dir):
    """
    Load image paths from the output directory.

    Args:
        output_dir (str): Path to the 'output_mnist' directory.

    Returns:
        image_info_list (list of dict): List containing image paths and labels.
    """
    image_info_list = []
    label_names = os.listdir(output_dir)
    label_names = [label for label in label_names if os.path.isdir(os.path.join(output_dir, label))]

    for label in label_names:
        label_dir = os.path.join(output_dir, label)
        annotated_dir = os.path.join(label_dir, "annotated_images")
        csv_dir = os.path.join(label_dir, "csv_files")

        # List all annotated images
        annotated_images = sorted(os.listdir(annotated_dir))
        csv_files = sorted(os.listdir(csv_dir))

        for annotated_img, csv_file in zip(annotated_images, csv_files):
            img_num = annotated_img.split('_')[1].split('.')[0]  # Extracting number from 'annotated_XXX.png'
            original_img = f"original_{img_num}.png"
            original_path = os.path.join(label_dir, "original_images", original_img)
            annotated_path = os.path.join(label_dir, "annotated_images", annotated_img)
            csv_path = os.path.join(label_dir, "csv_files", csv_file)

            image_info_list.append({
                'label': label,
                'original_path': original_path,
                'annotated_path': annotated_path,
                'csv_path': csv_path
            })

    return image_info_list


# -----------------------------
# Main Inference and Visualization
# -----------------------------

def infer_and_visualize():
    # -----------------------------
    # Parameters (Adjust as Needed)
    # -----------------------------
    data_folder = '/Users/yahyaabdelhamed/Documents/tum-adlr-03/src/output_mnist'  # Path to preprocessed data
    model_path = 'trained_models/deepsdf_model.pth'  # Path to the trained model
    latent_codes_path = 'trained_models/latent_codes.npy'  # Path to latent codes
    grid_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_reconstructions = 6  # Number of shapes to reconstruct and visualize

    # -----------------------------
    # Load Latent Codes
    # -----------------------------
    latent_codes = load_latent_codes(latent_codes_path)  # Shape: (num_embeddings, latent_dim)
    print(f"Loaded latent codes with shape: {latent_codes.shape}")

    # -----------------------------
    # Load Image Information
    # -----------------------------
    image_info_list = load_image_paths(data_folder)
    print(f"Total processed images found: {len(image_info_list)}")

    if len(image_info_list) == 0:
        print("No processed images found. Ensure that the 'output_mnist' directory contains processed images.")
        return

    # -----------------------------
    # Initialize and Load the Model
    # -----------------------------
    # Define model parameters (ensure they match the trained model)
    latent_dim = 256
    hidden_dim = 512
    num_layers = 8
    num_embeddings = latent_codes.shape[0]  # Ensure this matches the number of latent codes

    model = DeepSDFModel(latent_dim=latent_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                         num_embeddings=num_embeddings)
    model.to(device)

    # Load the trained model weights
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please provide the correct path.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded trained model from {model_path}.")

    # -----------------------------
    # Select Random Shapes for Reconstruction
    # -----------------------------
    if latent_codes.shape[0] < num_reconstructions:
        print(f"Not enough latent codes to select {num_reconstructions} shapes.")
        return

    # **Modify Here:** Select random indices within the range of latent codes
    random_indices = random.sample(range(latent_codes.shape[0]), num_reconstructions)
    print(f"Selected image indices for reconstruction: {random_indices}")

    # -----------------------------
    # Generate Grid Points
    # -----------------------------
    grid_points, grid_x, grid_y = generate_grid(grid_size=grid_size)
    grid_points = grid_points.to(device)  # Move to device

    # -----------------------------
    # Inference and Visualization Loop
    # -----------------------------
    for idx, image_idx in enumerate(random_indices):
        image_info = image_info_list[image_idx]
        label = image_info['label']
        original_path = image_info['original_path']
        annotated_path = image_info['annotated_path']
        csv_path = image_info['csv_path']

        # Retrieve the latent code for the selected image
        latent_code = torch.tensor(latent_codes[image_idx]).unsqueeze(0).to(device)  # Shape: (1, latent_dim)

        # Expand latent code to match the number of grid points
        latent_codes_expanded = latent_code.repeat(grid_points.size(0), 1)  # Shape: (grid_size**2, latent_dim)

        # Predict SDF values using the decoder
        with torch.no_grad():
            sdf_values = model.decoder(latent_codes_expanded, grid_points)  # Shape: (grid_size**2, 1)
            sdf_grid = sdf_values.view(grid_size, grid_size).cpu().numpy()
        print(sdf_grid)
        # Plot the reconstructed borders
        title = f"Reconstructed Border - Label: {label}, Image Index: {image_idx}"
        plot_reconstructed_borders(sdf_grid, grid_x, grid_y, title)

    print("Inference and visualization complete.")


if __name__ == "__main__":
    infer_and_visualize()
