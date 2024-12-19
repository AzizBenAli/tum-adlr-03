from scripts.models.decoder import DeepSDFModel
from helper import *

if __name__ == "__main__":
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
    state_dict = torch.load('../../multiclass/trained_models/deepsdf_model.pth', map_location=torch.device(device))
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
        save_path='../../multi_class/plots/latent_interpolation_linear.gif'
    )
    print("Animation created and saved as 'latent_interpolation_sequence.gif'.")



