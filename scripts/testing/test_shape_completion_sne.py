import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import imageio

from helper import visualize_shape_with_latent_shape_completion
from scripts.data_manipulation.data_loader import DeepSDFDataset2D
from scripts.models.decoder import DeepSDFModel

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def plot_tsne_with_random_init(model, random_latent_code, save_path=None):
    random_latent_code_cpu = random_latent_code.detach().cpu()

    learned_latent_embeddings = model.latent_codes.weight.detach().cpu()
    all_embeddings = torch.cat([learned_latent_embeddings, random_latent_code_cpu], dim=0)
    all_embeddings_np = all_embeddings.numpy()
    tsne = PCA(n_components=2, random_state=42)
    all_embeddings_2d = tsne.fit_transform(all_embeddings_np)

    learned_2d = all_embeddings_2d[:-1]
    random_2d = all_embeddings_2d[-1]

    plt.figure(figsize=(8, 8))
    plt.scatter(learned_2d[:, 0], learned_2d[:, 1],
                alpha=0.7, label="Learned Latent Codes", color="blue")
    plt.scatter(random_2d[0], random_2d[1],
                s=120, color="red", label="Random Initialization")
    plt.title("t-SNE of Latent Space + Random Initialization")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

def infer_and_visualize_shape(
    model,
    test_dataset,
    test_data_loader,
    shape_idx,
    plots_dir,
    grid_size=224,
    grid_range=(-10, 10),
    device="cpu",
    num_iterations=500,
    lr=1e-2,
    initial_latent_code=None,
    video_name="latent_optimization.mp4"
):
    model.eval()

    video_images_dir = os.path.join(plots_dir, "latent_video")
    os.makedirs(video_images_dir, exist_ok=True)

    full_points = []
    full_sdf = []
    shape_count = 0

    for batch in test_data_loader:
        batch_shape_idx = batch["shape_idx"]
        if batch_shape_idx[0].item() == shape_idx:
            full_points.append(batch["point"].to(device).float())
            full_sdf.append(batch["sdf"].to(device).float())
            shape_count = batch["shape_count"][0]

    full_points = torch.cat(full_points, dim=0)
    full_sdf = torch.cat(full_sdf, dim=0)
    print("Full points shape:", full_points.shape)

    num_samples = full_points.shape[0]
    random_samples = num_samples // 100
    indices = torch.randperm(num_samples)[:random_samples]
    partial_points = full_points[indices]
    partial_sdf = full_sdf[indices]

    if initial_latent_code is not None:
        latent_code = initial_latent_code.clone().detach().to(device)
    else:
        latent_code = torch.normal(
            mean=0.0,
            std=0.01,
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

        if (iteration + 1) % 30 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.6f}")
            plot_tsne_with_random_init(
                model=model,
                random_latent_code=latent_code,
                save_path=os.path.join(video_images_dir, f"iteration_{iteration + 1}.png")
            )

    images = sorted(
        [os.path.join(video_images_dir, img) for img in os.listdir(video_images_dir)],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    with imageio.get_writer(os.path.join(plots_dir, video_name), fps=5) as writer:
        for image_path in images:
            writer.append_data(imageio.imread(image_path))

    visualize_shape_with_latent_shape_completion(
        model,
        test_dataset,
        latent_code,
        shape_count,
        plots_dir,
        grid_size,
        grid_range,
        device,
        partial_points
    )

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
        num_embeddings=num_embeddings
    )
    state_dict_path = "../../multi_class/trained_models/deepsdf_model.pth"
    state_dict = torch.load(state_dict_path, map_location=device)
    trained_model.load_state_dict(state_dict)
    trained_model.to(device)
    trained_model.eval()

    test_dataset = DeepSDFDataset2D("../../multi_class/data", split="test")
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

    plots_dir = "../../multi_class/plots"

    random_init_code = torch.normal(
        mean=-1.0,
        std=0.01,
        size=(1, trained_model.latent_dim)
    ).to(device)

    plot_tsne_with_random_init(
        model=trained_model,
        random_latent_code=random_init_code,
    )

    final_code = infer_and_visualize_shape(
        model=trained_model,
        test_dataset=test_dataset,
        test_data_loader=test_loader,
        shape_idx=2,
        plots_dir=plots_dir,
        grid_size=500,
        grid_range=(-448, 448),
        device=device,
        num_iterations=10000,
        lr=1e-1,
        initial_latent_code=random_init_code
    )

    plot_tsne_with_random_init(
         model=trained_model,
         random_latent_code=final_code,
    )
