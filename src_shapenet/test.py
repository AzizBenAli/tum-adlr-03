import matplotlib.pyplot as plt
import numpy as np
import torch
from model import DeepSDFModel
from torch.utils.data import DataLoader

def visualize_shape_with_latent(model, latent_code, grid_size=224, grid_range=(-10, 10), device='cpu'):
    """
    Visualizes a shape using a given latent code.

    Args:
        model: The trained DeepSDFModel.
        latent_code: The latent code to use for visualization.
        grid_size: The size of the image (grid_size x grid_size).
        grid_range: A tuple (min, max) specifying the range of the grid along each axis.
        device: The device to perform computations on ('cpu' or 'cuda').
    """
    grid_size = int(grid_size)
    model.eval()
    with torch.no_grad():
        # Create a grid of points
        x = np.linspace(grid_range[0], grid_range[1], grid_size)
        y = np.linspace(grid_range[0], grid_range[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # Shape: (grid_size^2, 2)
        points = torch.tensor(points, dtype=torch.float32).to(device)

        # Repeat the latent code for all points
        latent_code_expanded = latent_code.repeat(points.shape[0], 1)  # Shape: (grid_size^2, latent_dim)

        # Predict SDF values
        sdf_values = model.decoder(latent_code_expanded, points)  # Shape: (grid_size^2, 1)
        sdf_values = sdf_values.cpu().numpy().reshape((grid_size, grid_size))  # Shape: (grid_size, grid_size)

        # Rotate the image by 90 degrees clockwise if needed
        sdf_values = np.rot90(sdf_values, k=1)

        # Create an image where inside the shape is white (255), outside is black (0)
        image = np.zeros((grid_size, grid_size), dtype=np.uint8)  # Black background
        image[sdf_values < 0] = 255  # Set pixels inside the shape to white

        # Display the image
        plt.figure(figsize=(8, 8))
        plt.imshow(image, extent=(grid_range[0], grid_range[1], grid_range[0], grid_range[1]),
                   origin='lower', cmap='gray')
        plt.axis('equal')
        plt.title(f'Reconstructed Shape')
        plt.axis('off')
        plt.show()

def infer_and_visualize_shape(model, test_data_loader, shape_idx, grid_size=224, grid_range=(-10, 10), device='cpu', num_iterations=500, lr=1e-2):
    """
    Infers the latent code for a test shape by optimizing a random latent vector
    and visualizes the reconstructed shape.

    Args:
        model: The trained DeepSDFModel.
        test_data_loader: DataLoader for the test dataset.
        shape_idx: The index of the shape to infer and visualize.
        grid_size: The size of the image (grid_size x grid_size).
        grid_range: A tuple (min, max) specifying the range of the grid along each axis.
        device: The device to perform computations on ('cpu' or 'cuda').
        num_iterations: Number of optimization iterations.
        lr: Learning rate for the optimizer.
    """
    model.eval()
    # Extract the data for the specific shape
    for batch in test_data_loader:
        batch_shape_idx = batch['shape_idx']
        if batch_shape_idx[0].item() == shape_idx:
            points = batch['point'].to(device).float()
            sdf = batch['sdf'].to(device).float()
            break
    else:
        print(f"Shape index {shape_idx} not found in test data.")
        return

    # Initialize a random latent code
    latent_dim = model.latent_dim
    latent_code = torch.randn((1, latent_dim), requires_grad=True, device=device)

    # Set up the optimizer for the latent code
    optimizer = torch.optim.Adam([latent_code], lr=lr)

    # Optimization loop
    criterion = torch.nn.MSELoss()
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        # Repeat the latent code for all points
        latent_code_expanded = latent_code.repeat(points.shape[0], 1)
        # Predict SDF values
        predicted_sdf = model.decoder(latent_code_expanded, points)
        # Compute loss
        loss = criterion(predicted_sdf, sdf)
        # Backpropagation
        loss.backward()
        optimizer.step()

        # Optional: Print loss every 50 iterations
        if (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.6f}")

    # Use the optimized latent code to visualize the shape
    visualize_shape_with_latent(model, latent_code, grid_size, grid_range, device)



if __name__ == "__main__":
    # Define the same parameters used during training
    latent_dim = 64
    hidden_dim = 512
    num_layers = 16
    num_embeddings = 69  # Number of embeddings used during training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the model instance
    trained_model = DeepSDFModel(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_embeddings=num_embeddings
    )
    state_dict = torch.load('../trained_models/deepsdf_model.pth', map_location=torch.device(device))
    trained_model.load_state_dict(state_dict)
    trained_model.to(device)
    trained_model.eval()

    # Load the test dataset
    from data_loader import DeepSDFDataset2D
    test_dataset = DeepSDFDataset2D('../data_shapeNet', split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Infer and visualize a test shape
    test_shape_idx = 0  # Change this to the index of the test shape you want to process
    infer_and_visualize_shape(
        trained_model,
        test_loader,
        test_shape_idx,
        grid_size=500,
        grid_range=(-448, 448),
        device=device,
        num_iterations=15000,
        lr=1e-2
    )
