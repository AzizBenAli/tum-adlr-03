import matplotlib.pyplot as plt
import numpy as np
import torch
from model import DeepSDFModel

import matplotlib.pyplot as plt
import numpy as np
import torch
from model import DeepSDFModel

def visualize_shape(model, shape_idx, grid_size=224, grid_range=(-10, 10), device='cpu'):
    """
    Visualizes the shape corresponding to the given shape index using the trained model.

    Args:
        model: The trained DeepSDFModel.
        shape_idx: The index of the shape to visualize.
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

        # Get the latent code for the specified shape index from the model
        latent_code = model.latent_codes(torch.tensor([shape_idx], device=device))  # Shape: (1, latent_dim)
        # Repeat the latent code for all points
        latent_code_expanded = latent_code.repeat(points.shape[0], 1)  # Shape: (grid_size^2, latent_dim)

        # Predict SDF values
        sdf_values = model.decoder(latent_code_expanded, points)  # Shape: (grid_size^2, 1)
        sdf_values = sdf_values.cpu().numpy().reshape((grid_size, grid_size))  # Shape: (grid_size, grid_size)

        # Optionally, print the min and max of sdf_values
        print(f"SDF min: {sdf_values.min()}, SDF max: {sdf_values.max()}")

        # Rotate the image by 90 degrees clockwise
        sdf_values = np.rot90(sdf_values, k=1)

        # Create an image where inside the shape is white (255), outside is black (0)
        image = np.zeros((grid_size, grid_size), dtype=np.uint8)  # Black background
        image[sdf_values < 0] = 255  # Set pixels inside the shape to white

        # Display the image
        plt.figure(figsize=(8, 8))
        plt.imshow(image, extent=(grid_range[0], grid_range[1], grid_range[0], grid_range[1]),
                   origin='lower', cmap='gray')
        plt.axis('equal')
        plt.title(f'Shape Index: {shape_idx}')
        plt.axis('off')
        plt.show()



# Assuming 'trained_model' is your trained DeepSDFModel
# and 'train_dataset' is your training dataset
# Define the same parameters used during training
# Define the same parameters used during training
latent_dim = 64
hidden_dim = 512
num_layers = 16
num_embeddings = 69  # Adjust based on your number of shapes
device = "cpu"

# Create the model instance
trained_model = DeepSDFModel(
    latent_dim=latent_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_embeddings=num_embeddings
)
state_dict = torch.load('../trained_models/deepsdf_model.pth', map_location=torch.device(device))

# Load the state dictionary into the model
trained_model.load_state_dict(state_dict)
trained_model.to(device)
trained_model.eval()

# Visualize the shapes
for shape_idx in range(20):  # Assuming you have at least 6 shapes
    visualize_shape(trained_model, shape_idx, grid_size=500, grid_range=(-448, 448), device=device)


