import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scripts.models.decoder import DeepSDFModel
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from scripts.data_transformation.data_loader import DeepSDFDataset2D

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


"""if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 128
    hidden_dim = 512
    num_layers = 32
    num_embeddings = 630
    grid_size = 500
    grid_range = (-448, 448)

    print("Loading the trained model...")
    trained_model = DeepSDFModel(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_embeddings=num_embeddings
    )
    state_dict = torch.load('../../multiclass/trained_models/trained_decoder/deepsdf_model.pth', map_location=torch.device(device))
    trained_model.load_state_dict(state_dict)
    trained_model.to(device)
    trained_model.eval()
    print("Model loaded successfully.")

    print("Visualizing the latent space using t-SNE...")
    visualize_latent_space(trained_model, device=device)
    print("Latent space visualization complete.")

    shape_indices = list(range(1, 11))
    print(f"Interpolating through shapes: {shape_indices}")

    latent_codes = []
    for idx in shape_indices:
        latent_code = trained_model.latent_codes(torch.tensor([idx], device=device))
        latent_codes.append(latent_code)
        print(f"Shape index {idx} corresponds to shape {search_train_dataset(idx)[0]}")

    num_steps_per_pair = 50
    total_interpolated_latents = []

    for i in range(len(latent_codes) - 1):
        print(f"Interpolating between shape {shape_indices[i]} and shape {shape_indices[i+1]}...")
        interpolated_latents = interpolate_latent_codes(
            latent_codes[i],
            latent_codes[i+1],
            num_steps=num_steps_per_pair
        )

        if i < len(latent_codes) - 2:
            interpolated_latents = interpolated_latents[:-1]
        total_interpolated_latents.extend(interpolated_latents)

    total_interpolated_latents.append(latent_codes[-1])

    print("Interpolation between all shapes complete.")

    print("Creating animation for the sequence of interpolated shapes...")
    anim_sequence = create_interpolation_animation(
        model=trained_model,
        interpolated_latents=total_interpolated_latents,
        grid_size=grid_size,
        grid_range=grid_range,
        device=device,
        save_path='../plots/latent_interpolation_sequence.gif'
    )
    print("Animation created and saved as 'latent_interpolation_sequence.gif'.")

exit()"""

if __name__ == "__main__":
    device = "mps" if torch.cuda.is_available() else "cpu"
    latent_dim = 64
    hidden_dim = 512
    num_layers = 16
    num_embeddings = 70
    grid_size = 500
    grid_range = (-448, 448)

    print("Loading the trained model...")
    trained_model = DeepSDFModel(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_embeddings=num_embeddings
    )
    state_dict = torch.load('../../multi_class/trained_models/deepsdf_model.pth', map_location=torch.device(device))
    trained_model.load_state_dict(state_dict)
    trained_model.to(device)
    trained_model.eval()
    print("Model loaded successfully.")

    print("Visualizing the latent space using t-SNE...")
    visualize_latent_space(trained_model, device=device)
    print("Latent space visualization complete.")

    shape_idx_1 = 1
    print(f"Shape index {shape_idx_1} corresponds to shape {search_train_dataset(shape_idx_1)}")
    shape_idx_2 = 23
    print(f"Shape index {shape_idx_2} corresponds to shape {search_train_dataset(shape_idx_2)}")
    print(f"Selecting latent codes for shapes {shape_idx_1} and {shape_idx_2}.")

    latent_code_1 = trained_model.latent_codes(torch.tensor([shape_idx_1], device=device))
    latent_code_2 = trained_model.latent_codes(torch.tensor([shape_idx_2], device=device))

    num_steps = 50
    print("Performing linear interpolation between latent codes...")
    interpolated_latents_linear = interpolate_latent_codes(latent_code_1, latent_code_2, num_steps=num_steps)
    print("Linear interpolation complete.")

    print("Creating animation for linear interpolation...")
    anim_linear = create_interpolation_animation(
        model=trained_model,
        interpolated_latents=interpolated_latents_linear,
        grid_size=grid_size,
        grid_range=grid_range,
        device=device,
        save_path='../../multi_class/plots/latent_interpolation_linear.gif'
    )
    print("Linear interpolation animation created and saved as 'latent_interpolation_linear.gif'.")

    print("Performing spherical linear interpolation between latent codes...")
    interpolated_latents_slerp = slerp(latent_code_1, latent_code_2, num_steps=num_steps)
    print("Spherical linear interpolation complete.")

    print("Creating animation for spherical linear interpolation...")
    anim_slerp = create_interpolation_animation(
        model=trained_model,
        interpolated_latents=interpolated_latents_slerp,
        grid_size=grid_size,
        grid_range=grid_range,
        device=device,
        save_path='../../multi_class/plots/latent_interpolation_slerp.gif'
    )
    print("Spherical linear interpolation animation created and saved as 'latent_interpolation_slerp.gif'.")

    print("All tasks completed successfully.")

