import torch
import numpy as np
import matplotlib.pyplot as plt
from model import DeepSDFModel
from torch.utils.data import DataLoader
from src.output.Data_loader import DeepSDFDataset2D


def optimize_latent_code(model, points, sdf, latent_code, lr=1e-3, num_iters=100):
    """
    Optimize the latent code for an unseen shape during testing.
    """
    latent_code = latent_code.clone().detach().requires_grad_(True)  # Enable gradients for optimization
    optimizer = torch.optim.Adam([latent_code], lr=lr)

    for _ in range(num_iters):
        optimizer.zero_grad()
        predicted_sdf = model.decoder(latent_code.repeat(points.size(0), 1), points)
        loss = torch.nn.functional.l1_loss(predicted_sdf, sdf)
        loss.backward()
        optimizer.step()

    return latent_code


def generate_grid(grid_size=128, range_min=-1.0, range_max=1.0):
    """
    Generate a 2D grid of points covering the image space.

    Args:
        grid_size (int): Resolution of the grid.
        range_min (float): Minimum coordinate value.
        range_max (float): Maximum coordinate value.

    Returns:
        grid_points (torch.Tensor): Tensor of shape (grid_size**2, 2).
    """
    x = np.linspace(range_min, range_max, grid_size)
    y = np.linspace(range_min, range_max, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    return torch.tensor(grid_points, dtype=torch.float32), grid_x, grid_y


def plot_sdf_points(points, sdf_values, title="SDF Values"):
    """
    Plot SDF values at their corresponding (x, y) positions.

    Args:
        points (numpy.ndarray): Array of shape (N, 2) containing (x, y) positions.
        sdf_values (numpy.ndarray): Array of shape (N,) containing SDF values.
        title (str): Title of the plot.
    """
    sdf_values = sdf_values.flatten()  # Ensure sdf_values is 1D

    plt.figure(figsize=(8, 8))

    # Separate points based on SDF value sign
    negative_points = points[sdf_values < 0]  # Inside points (green)
    positive_points = points[sdf_values >= 0]  # Outside points (blue)

    # Plot inside points
    plt.scatter(negative_points[:, 0], negative_points[:, 1], c='green', label='Inside (SDF < 0)', s=5)

    # Plot outside points
    plt.scatter(positive_points[:, 0], positive_points[:, 1], c='blue', label='Outside (SDF >= 0)', s=5)

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    # Load test dataset
    data_folder = '/Users/yahyaabdelhamed/Documents/tum-adlr-03/src/output_mnist'
    test_dataset = DeepSDFDataset2D(data_folder, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    latent_dim = 256
    num_embeddings = 700  # Number of shapes in training
    model = DeepSDFModel(latent_dim=latent_dim, num_embeddings=num_embeddings)
    model.load_state_dict(torch.load('trained_models/deepsdf_model.pth', map_location=device))
    model.to(device)

    # Reconstruct an image from the test set
    for i, batch in enumerate(test_loader):
        if i >= 1:  # Reconstruct only one image for demonstration
            break

        points = batch['point'].to(device)
        sdf = batch['sdf'].to(device)

        # Randomly initialize latent code for unseen shape
        latent_code = torch.randn(1, latent_dim, device=device)
        latent_code = optimize_latent_code(model, points, sdf, latent_code)

        # Generate grid and predict SDF for the entire grid
        grid_points, grid_x, grid_y = generate_grid(grid_size=128)  # Higher grid size for better visualization
        grid_points = grid_points.to(device)

        with torch.no_grad():
            predicted_sdf_grid = model.decoder(latent_code.repeat(grid_points.size(0), 1), grid_points).cpu().numpy()

        # Plot SDF values for all grid points
        plot_sdf_points(grid_points.cpu().numpy(), predicted_sdf_grid, title=f"Predicted SDF for Shape {i + 1}")
