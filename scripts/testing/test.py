import torch
from helper import visualize_shape_with_latent
from scripts.data_manipulation.data_loader import DeepSDFDataset2D
from torch.utils.data import DataLoader
from scripts.models.decoder import DeepSDFModel

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def infer_and_visualize_shape(model, test_dataset, test_data_loader, shape_idx, plots_dir, grid_size=224, grid_range=(-10, 10), device='cpu', num_iterations=500, lr=1e-2):
    model.eval()
    for batch in test_data_loader:
        batch_shape_idx = batch['shape_idx']
        if batch_shape_idx[0].item() == shape_idx:
            points = batch['point'].to(device).float()
            sdf = batch['sdf'].to(device).float()
            break
    else:
        print(f"Shape index {shape_idx} not found in test data.")
        return

    latent_dim = model.latent_dim
    latent_code = torch.randn((1, latent_dim), requires_grad=True, device=device)

    optimizer = torch.optim.Adam([latent_code], lr=lr)

    criterion = torch.nn.MSELoss()
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        latent_code_expanded = latent_code.repeat(points.shape[0], 1)
        predicted_sdf = model.decoder(latent_code_expanded, points)
        loss = criterion(predicted_sdf, sdf)
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.6f}")

    visualize_shape_with_latent(model, test_dataset, latent_code, batch['shape_count'], plots_dir, grid_size, grid_range, device)









