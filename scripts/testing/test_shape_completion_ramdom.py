import torch
from helper import visualize_shape_with_latent_shape_completion
from scripts.data_manipulation.data_loader import DeepSDFDataset2D
from torch.utils.data import DataLoader
from scripts.models.decoder import DeepSDFModel

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    lr=1e-2
):

    model.eval()

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
    print(full_points.shape)
    full_sdf = torch.cat(full_sdf, dim=0)

    num_samples = full_points.shape[0]
    half_samples = num_samples // 30
    indices = torch.randperm(num_samples)[:half_samples]
    partial_points = full_points[indices]
    partial_sdf = full_sdf[indices]

    latent_mean = 0.0
    latent_std = 0.01
    latent_code = torch.normal(
        mean=latent_mean,
        std=latent_std,
        size=(1, model.latent_dim),
        device=device,
        requires_grad=True
    )

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
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.6f}")

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

if __name__ == "__main__":
    latent_dim = 64
    hidden_dim = 512
    num_layers = 32
    num_embeddings = 314

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trained_model = DeepSDFModel(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_embeddings=num_embeddings
    )
    state_dict_path = "../../multi_class/trained_models/deepsdf_model.pth"
    state_dict = torch.load(state_dict_path, map_location=torch.device(device))
    trained_model.load_state_dict(state_dict)
    trained_model.to(device)
    trained_model.eval()

    test_dataset = DeepSDFDataset2D("../../multi_class/data", split="test")
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

    plots_dir = "../../multi_class/plots"

    infer_and_visualize_shape(
        trained_model,
        test_dataset,
        test_loader,
        shape_idx=19,
        plots_dir=plots_dir,
        grid_size=500,
        grid_range=(-448, 448),
        device=device,
        num_iterations=2600,
        lr=1e-1,
    )
