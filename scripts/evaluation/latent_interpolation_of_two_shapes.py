from scripts.models.decoder import DeepSDFModel
import torch
from scripts.helpers.visualization_helpers import visualize_latent_space
from scripts.helpers.latent_helpers import (
    search_train_dataset,
    slerp,
    interpolate_latent_codes,
)
from scripts.utils.image_utils import create_interpolation_animation

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 64
    hidden_dim = 512
    num_layers = 32
    num_embeddings = 314
    grid_size = 500
    grid_range = (-448, 448)

    print("Loading the trained model...")
    trained_model = DeepSDFModel(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_embeddings=num_embeddings,
    )
    state_dict = torch.load(
        "../../multi_class/trained_models/deepsdf_model.pth",
        map_location=torch.device(device),
    )
    trained_model.load_state_dict(state_dict)
    trained_model.to(device)
    trained_model.eval()
    print("Model loaded successfully.")

    print("Visualizing the latent space using t-SNE...")
    visualize_latent_space(trained_model, device=device)
    print("Latent space visualization complete.")

    shape_idx_1 = 1
    print(
        f"Shape index {shape_idx_1} corresponds to shape {search_train_dataset(shape_idx_1)}"
    )
    shape_idx_2 = 65
    print(
        f"Shape index {shape_idx_2} corresponds to shape {search_train_dataset(shape_idx_2)}"
    )
    print(f"Selecting latent codes for shapes {shape_idx_1} and {shape_idx_2}.")

    latent_code_1 = trained_model.latent_codes(
        torch.tensor([shape_idx_1], device=device)
    )
    latent_code_2 = trained_model.latent_codes(
        torch.tensor([shape_idx_2], device=device)
    )

    num_steps = 50
    print("Performing linear interpolation between latent codes...")
    interpolated_latents_linear = interpolate_latent_codes(
        latent_code_1, latent_code_2, num_steps=num_steps
    )
    print("Linear interpolation complete.")

    print("Creating animation for linear interpolation...")
    anim_linear = create_interpolation_animation(
        model=trained_model,
        interpolated_latents=interpolated_latents_linear,
        grid_size=grid_size,
        grid_range=grid_range,
        device=device,
        save_path="../../multi_class/visualization/latent_interpolation_linear.gif",
    )
    print(
        "Linear interpolation animation created and saved as 'latent_interpolation_linear.gif'."
    )

    print("Performing spherical linear interpolation between latent codes...")
    interpolated_latents_slerp = slerp(
        latent_code_1, latent_code_2, num_steps=num_steps
    )
    print("Spherical linear interpolation complete.")

    print("Creating animation for spherical linear interpolation...")
    anim_slerp = create_interpolation_animation(
        model=trained_model,
        interpolated_latents=interpolated_latents_slerp,
        grid_size=grid_size,
        grid_range=grid_range,
        device=device,
        save_path="../../multi_class/visualization/latent_interpolation_slerp.gif",
    )
    print(
        "Spherical linear interpolation animation created and saved as 'latent_interpolation_slerp.gif'."
    )

    print("All tasks completed successfully.")
