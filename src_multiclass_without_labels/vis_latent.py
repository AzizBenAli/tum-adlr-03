import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from model import DeepSDFModel  # Ensure you have the model definition in model.py
import os
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from data_loader import DeepSDFDataset2D

def interpolate_latent_codes(z1, z2, num_steps=10):
    """
    Performs linear interpolation between two latent codes.

    Args:
        z1: The first latent code (Tensor of shape [1, latent_dim]).
        z2: The second latent code (Tensor of shape [1, latent_dim]).
        num_steps: Number of interpolation steps.

    Returns:
        List of interpolated latent codes.
    """
    interpolated_latents = []
    for t in np.linspace(0, 1, num_steps):
        z_t = (1 - t) * z1 + t * z2
        interpolated_latents.append(z_t)
    return interpolated_latents

def slerp(z1, z2, num_steps=10):
    """
    Performs spherical linear interpolation (slerp) between two latent codes.

    Args:
        z1: The first latent code (Tensor of shape [1, latent_dim]).
        z2: The second latent code (Tensor of shape [1, latent_dim]).
        num_steps: Number of interpolation steps.

    Returns:
        List of interpolated latent codes.
    """
    z1_norm = z1 / torch.norm(z1)
    z2_norm = z2 / torch.norm(z2)
    omega = torch.acos(torch.clamp(torch.dot(z1_norm.squeeze(), z2_norm.squeeze()), -1.0 + 1e-7, 1.0 - 1e-7))
    sin_omega = torch.sin(omega)
    interpolated_latents = []
    for t in np.linspace(0, 1, num_steps):
        if sin_omega == 0:
            z_t = (1 - t) * z1 + t * z2  # Fallback to linear interpolation
        else:
            z_t = (torch.sin((1 - t) * omega) / sin_omega) * z1 + (torch.sin(t * omega) / sin_omega) * z2
        interpolated_latents.append(z_t)
    return interpolated_latents

def generate_shape_image(model, latent_code, grid_size=224, grid_range=(-10, 10), device='cpu'):
    """
    Generates a shape image from a latent code.

    Args:
        model: The trained DeepSDFModel.
        latent_code: The latent code to use for visualization.
        grid_size: The size of the image (grid_size x grid_size).
        grid_range: The range of the grid.
        device: The device to perform computations on.

    Returns:
        image: The generated image as a NumPy array.
    """
    grid_size = int(grid_size)
    model.eval()
    with torch.no_grad():
        # Create a grid of points
        x = np.linspace(grid_range[0], grid_range[1], grid_size)
        y = np.linspace(grid_range[0], grid_range[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([yy.ravel(),xx.ravel()], axis=-1)
        points = torch.tensor(points, dtype=torch.float32).to(device)

        # Repeat the latent code for all points
        latent_code_expanded = latent_code.repeat(points.shape[0], 1)

        # Predict SDF values
        sdf_values = model.decoder(latent_code_expanded, points)
        sdf_values = sdf_values.cpu().numpy().reshape((grid_size, grid_size))

        # Rotate the image by 90 degrees clockwise if needed
        #sdf_values = np.rot90(sdf_values, k=2)

        # Create an image where inside the shape is white (255), outside is black (0)
        image = np.zeros((grid_size, grid_size), dtype=np.uint8)
        image[sdf_values < 0] = 255

    return image

def create_interpolation_animation(model, interpolated_latents, grid_size=224, grid_range=(-10, 10),
                                   device='cpu', save_path=None):
    """
    Creates an animation of interpolated shapes.

    Args:
        model: The trained DeepSDFModel.
        interpolated_latents: List of interpolated latent codes.
        grid_size: The size of the image.
        grid_range: The range of the grid.
        device: The device to perform computations on.
        save_path: Path to save the animation (e.g., 'interpolation.gif').

    Returns:
        anim: The animation object.
    """
    # Generate images for each interpolated latent code
    images = []
    for z in interpolated_latents:
        image = generate_shape_image(model, z, grid_size, grid_range, device)
        images.append(image)

    # Create the animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    img_display = ax.imshow(images[0], cmap='gray', animated=True)

    def update(frame):
        img_display.set_array(images[frame])
        return [img_display]

    anim = FuncAnimation(fig, update, frames=len(images), interval=200, blit=True)

    # Save the animation if a save path is provided
    if save_path:
        anim.save(save_path, writer='pillow', fps=5)

    plt.close(fig)  # Close the figure to prevent it from displaying immediately
    return anim

def visualize_latent_space(model, device='cpu'):
    """
    Visualizes the latent space using t-SNE.

    Args:
        model: The trained DeepSDFModel.
        device: The device to perform computations on.
    """
    # Get all latent codes from the model
    latent_codes = model.latent_codes.weight.detach().cpu().numpy()  # Shape: (num_embeddings, latent_dim)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    latent_codes_2d = tsne.fit_transform(latent_codes)

    # Plot the 2D latent codes
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_codes_2d[:, 0], latent_codes_2d[:, 1], alpha=0.7)
    plt.title('Latent Space Visualization using t-SNE')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

def search_train_dataset(shape_idx):
    train_dataset = DeepSDFDataset2D('../data_shapeNet', split='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    for batch in train_loader:
        batch_shape_idx = batch['shape_idx']
        if batch_shape_idx[0].item() == shape_idx:
            return batch['shape_count']
    else:
        print(f"Shape index {shape_idx} not found in test data.")
        return


if __name__ == "__main__":
    """
    Main script to visualize the latent space of the trained model, perform interpolation
    between latent codes, and create animations demonstrating shape transformations.
    """
    # Step 1: Set up the device and model parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 64
    hidden_dim = 512
    num_layers = 16
    num_embeddings = 209  # Adjust based on your trained model
    grid_size = 500
    grid_range = (-448, 448)

    # Step 2: Load the trained DeepSDF model
    print("Loading the trained model...")
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
    print("Model loaded successfully.")

    # Step 3: Visualize the latent space using t-SNE
    print("Visualizing the latent space using t-SNE...")
    visualize_latent_space(trained_model, device=device)
    print("Latent space visualization complete.")

    # Step 4: Interpolate from shape 1 to shape 10 sequentially
    shape_indices = list(range(1, 11))  # Shape indices from 1 to 10
    print(f"Interpolating through shapes: {shape_indices}")

    # Retrieve latent codes for all shapes
    latent_codes = []
    for idx in shape_indices:
        latent_code = trained_model.latent_codes(torch.tensor([idx], device=device))
        latent_codes.append(latent_code)
        print(f"Shape index {idx} corresponds to shape {search_train_dataset(idx)[0]}")

    # Step 5: Interpolate between each consecutive pair and collect all interpolated latent codes
    num_steps_per_pair = 50  # Number of interpolation steps between each pair
    total_interpolated_latents = []

    for i in range(len(latent_codes) - 1):
        print(f"Interpolating between shape {shape_indices[i]} and shape {shape_indices[i+1]}...")
        # Perform interpolation between latent_codes[i] and latent_codes[i+1]
        interpolated_latents = interpolate_latent_codes(
            latent_codes[i],
            latent_codes[i+1],
            num_steps=num_steps_per_pair
        )
        # Exclude the last latent code to avoid duplicates except for the final shape
        if i < len(latent_codes) - 2:
            interpolated_latents = interpolated_latents[:-1]
        total_interpolated_latents.extend(interpolated_latents)

    # Append the last shape's latent code
    total_interpolated_latents.append(latent_codes[-1])

    print("Interpolation between all shapes complete.")

    # Step 6: Create an animation for the entire sequence
    print("Creating animation for the sequence of interpolated shapes...")
    anim_sequence = create_interpolation_animation(
        model=trained_model,
        interpolated_latents=total_interpolated_latents,
        grid_size=grid_size,
        grid_range=grid_range,
        device=device,
        save_path='../plots_single_class/latent_interpolation_sequence.gif'
        # Change or set to None if you don't want to save
    )
    print("Animation created and saved as 'latent_interpolation_sequence.gif'.")

    # Optional: Display the animation in a Jupyter Notebook
    # If you're running this script in a notebook, uncomment the following lines:

    # from IPython.display import HTML
    # print("Displaying the sequence interpolation animation:")
    # display(HTML(anim_sequence.to_jshtml()))

    print("All tasks completed successfully.")
    exit()

exit()
if __name__ == "__main__":
    """
    Main script to visualize the latent space of the trained model, perform interpolation
    between latent codes, and create animations demonstrating shape transformations.
    """
    # Step 1: Set up the device and model parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 64
    hidden_dim = 512
    num_layers = 16
    num_embeddings = 209  # Adjust based on your trained model
    grid_size = 500
    grid_range = (-448, 448)

    # Step 2: Load the trained DeepSDF model
    print("Loading the trained model...")
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
    print("Model loaded successfully.")

    # Step 3: Visualize the latent space using t-SNE
    print("Visualizing the latent space using t-SNE...")
    visualize_latent_space(trained_model, device=device)
    print("Latent space visualization complete.")

    # Step 4: Select two latent codes for interpolation
    # Choose two different shape indices from your dataset
    shape_idx_1 = 1  # Replace with actual shape index
    print(f"Shape index {shape_idx_1} corresponds to shape {search_train_dataset(shape_idx_1)}")
    shape_idx_2 = 8  # Replace with actual shape index
    print(f"Shape index {shape_idx_2} corresponds to shape {search_train_dataset(shape_idx_2)}")
    print(f"Selecting latent codes for shapes {shape_idx_1} and {shape_idx_2}.")

    # Retrieve the latent codes from the model's latent embeddings
    latent_code_1 = trained_model.latent_codes(torch.tensor([shape_idx_1], device=device))
    latent_code_2 = trained_model.latent_codes(torch.tensor([shape_idx_2], device=device))

    # Step 5: Perform linear interpolation between the latent codes
    num_steps = 50  # Number of interpolation steps
    print("Performing linear interpolation between latent codes...")
    interpolated_latents_linear = interpolate_latent_codes(latent_code_1, latent_code_2, num_steps=num_steps)
    print("Linear interpolation complete.")

    # Step 6: Create an animation for linear interpolation
    print("Creating animation for linear interpolation...")
    anim_linear = create_interpolation_animation(
        model=trained_model,
        interpolated_latents=interpolated_latents_linear,
        grid_size=grid_size,
        grid_range=grid_range,
        device=device,
        save_path='../plots_single_class/latent_interpolation_linear.gif'
        # Change or set to None if you don't want to save
    )
    print("Linear interpolation animation created and saved as 'latent_interpolation_linear.gif'.")

    # Step 7: Perform spherical linear interpolation (slerp) between the latent codes
    print("Performing spherical linear interpolation between latent codes...")
    interpolated_latents_slerp = slerp(latent_code_1, latent_code_2, num_steps=num_steps)
    print("Spherical linear interpolation complete.")

    # Step 8: Create an animation for spherical linear interpolation
    print("Creating animation for spherical linear interpolation...")
    anim_slerp = create_interpolation_animation(
        model=trained_model,
        interpolated_latents=interpolated_latents_slerp,
        grid_size=grid_size,
        grid_range=grid_range,
        device=device,
        save_path='../plots_single_class/latent_interpolation_slerp.gif'
        # Change or set to None if you don't want to save
    )
    print("Spherical linear interpolation animation created and saved as 'latent_interpolation_slerp.gif'.")

    # Optional: Display the animations in a Jupyter Notebook
    # If you're running this script in a notebook, uncomment the following lines:

    # from IPython.display import HTML
    # print("Displaying linear interpolation animation:")
    # display(HTML(anim_linear.to_jshtml()))
    # print("Displaying spherical linear interpolation animation:")
    # display(HTML(anim_slerp.to_jshtml()))

    print("All tasks completed successfully.")

