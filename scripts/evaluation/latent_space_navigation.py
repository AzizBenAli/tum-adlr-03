import os
import torch
from torch.utils.data import DataLoader
import imageio
from scripts.data_manipulation.data_loader import DeepSDFDataset2D
from scripts.models.decoder import DeepSDFModel
from scripts.helpers.visualization_helpers import (
    visualize_reconstructed_shape,
    plot_tsne_with_random_init,
)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def infer_and_navigate_in_latent_space(
    model,
    test_data_loader,
    shape_idx,
    plots_dir,
    grid_size=224,
    grid_range=(-10, 10),
    device="cpu",
    num_iterations=500,
    lr=1e-2,
    initial_latent_code=None,
    video_name_latent="videos/latent_optimization_train.mp4",
    video_name_object="videos/object_reconstruction_train.mp4",
):
    model.eval()

    video_dir = os.path.join(plots_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    navigation_latent_dir = os.path.join(plots_dir, "videos/latent_video")
    os.makedirs(navigation_latent_dir, exist_ok=True)

    navigation_object_dir = os.path.join(plots_dir, "videos/object_video")
    os.makedirs(navigation_object_dir, exist_ok=True)

    full_points = []
    full_sdf = []

    for batch in test_data_loader:
        batch_shape_idx = batch["shape_idx"]
        if batch_shape_idx[0].item() == shape_idx:
            full_points.append(batch["point"].to(device).float())
            full_sdf.append(batch["sdf"].to(device).float())

    full_points = torch.cat(full_points, dim=0)
    full_sdf = torch.cat(full_sdf, dim=0)
    print("Full points shape:", full_points.shape)

    num_samples = full_points.shape[0]
    random_samples = num_samples // 30
    indices = torch.randperm(num_samples)[:random_samples]
    partial_points = full_points[indices]
    partial_sdf = full_sdf[indices]

    if initial_latent_code is not None:
        latent_code = initial_latent_code.clone().detach().to(device)
    else:
        latent_code = torch.normal(
            mean=0.0,
            std=1,
            size=(1, model.latent_dim),
        ).to(device)

    latent_code.requires_grad = True

    optimizer = torch.optim.Adam([latent_code], lr=lr)
    criterion = torch.nn.MSELoss()

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        latent_code_expanded = latent_code.repeat(partial_points.shape[0], 1)
        predicted_sdf = model.decoder(latent_code_expanded, partial_points)
        loss = criterion(predicted_sdf, partial_sdf)
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 50 == 0:
            print(
                f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.6f}"
            )
            plot_tsne_with_random_init(
                model=model,
                random_latent_code=latent_code,
                save_path=os.path.join(
                    navigation_latent_dir, f"iteration_{iteration + 1}.png"
                ),
            )
            visualize_reconstructed_shape(
                model,
                latent_code,
                plots_dir,
                grid_size,
                grid_range,
                device,
                save_path=os.path.join(
                    navigation_object_dir, f"iteration_{iteration + 1}.png"
                ),
            )

    images_latent = sorted(
        [
            os.path.join(navigation_latent_dir, img)
            for img in os.listdir(navigation_latent_dir)
        ],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    images_shape = sorted(
        [
            os.path.join(navigation_object_dir, img)
            for img in os.listdir(navigation_object_dir)
        ],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    with imageio.get_writer(
        os.path.join(plots_dir, video_name_latent), fps=5
    ) as writer:
        for image_path in images_latent:
            writer.append_data(imageio.imread(image_path))

    with imageio.get_writer(
        os.path.join(plots_dir, video_name_object), fps=5
    ) as writer:
        for image_path in images_shape:
            writer.append_data(imageio.imread(image_path))

    return latent_code


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 64
    hidden_dim = 512
    num_layers = 32
    num_embeddings = 314

    trained_model = DeepSDFModel(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_embeddings=num_embeddings,
    )
    state_dict_path = "../../multi_class/trained_models/deepsdf_model.pth"
    state_dict = torch.load(state_dict_path, map_location=device)
    trained_model.load_state_dict(state_dict)
    trained_model.to(device)
    trained_model.eval()

    test_dataset = DeepSDFDataset2D("../../multi_class/data", split="test")
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

    plots_dir = "../../multi_class/visualization"

    random_init_code = torch.normal(
        mean=0.0, std=0.01, size=(1, trained_model.latent_dim)
    ).to(device)

    plot_tsne_with_random_init(
        model=trained_model,
        random_latent_code=random_init_code,
    )

    final_code = infer_and_navigate_in_latent_space(
        model=trained_model,
        test_data_loader=test_loader,
        shape_idx=0,
        plots_dir=plots_dir,
        grid_size=500,
        grid_range=(-448, 448),
        device=device,
        num_iterations=1000,
        lr=1e-1,
        initial_latent_code=random_init_code,
    )

    plot_tsne_with_random_init(
        model=trained_model,
        random_latent_code=final_code,
    )
