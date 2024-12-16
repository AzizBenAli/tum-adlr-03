import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from model import DeepSDFModel
from torch.utils.data import DataLoader
from data_loader import DeepSDFDataset2D

def visualize_shape_with_latent(model, latent_code, label, shape_count, grid_size=224, grid_range=(-10, 10), device='cpu'):
    grid_size = int(grid_size)
    model.eval()
    with torch.no_grad():
        x = np.linspace(grid_range[0], grid_range[1], grid_size)
        y = np.linspace(grid_range[0], grid_range[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        points = torch.tensor(points, dtype=torch.float32).to(device)

        latent_code_expanded = latent_code.repeat(points.shape[0], 1)
        label_expanded = torch.full((points.shape[0], 1), label, dtype=torch.float32, device=device)

        inputs = torch.cat([latent_code_expanded, label_expanded, points], dim=1)
        sdf_values = model.decoder(inputs)
        sdf_values = sdf_values.cpu().numpy().reshape((grid_size, grid_size))

        sdf_values = np.rot90(sdf_values, k=1)
        image = np.zeros((grid_size, grid_size), dtype=np.uint8)
        image[sdf_values < 0] = 255

        shape_count = int(shape_count[0])
        annotated_image_path = f"shape{shape_count}_annotated.png"
        image_dir = os.path.join(test_dataset.root_folder, "cars", 'annotated_images')
        image_path = os.path.join(image_dir, annotated_image_path)
        annotated_image = plt.imread(image_path)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(image, extent=(grid_range[0], grid_range[1], grid_range[0], grid_range[1]),
                       origin='lower', cmap='gray')
        axes[0].set_title('Reconstructed Shape')
        axes[0].axis('off')
        axes[0].set_aspect('equal')

        axes[1].imshow(annotated_image, cmap='gray')
        axes[1].set_title('Annotated Shape')
        axes[1].axis('off')
        axes[1].set_aspect('equal')

        plt.tight_layout()
        os.makedirs("../plots", exist_ok=True)
        plt.savefig(f"../plots/reconstructed_image_{shape_count}.png")
        plt.show()


def infer_and_visualize_shape(model, test_data_loader, shape_idx, grid_size=224, grid_range=(-10, 10), device='cpu', num_iterations=500, lr=1e-2):
    model.eval()
    for batch in test_data_loader:
        batch_shape_idx = batch['shape_idx']
        if batch_shape_idx[0].item() == shape_idx:
            points = batch['point'].to(device).float()
            sdf = batch['sdf'].to(device).float()
            label_indices = batch['label_idx'].to(device).long()
            label = batch['label_idx'][0].to(device).item()
            shape_count = batch['shape_count']
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
        #label_expanded = torch.full((points.shape[0], 1), label, dtype=torch.long, device=device)
        #scale label needed
        label_scaled = label_indices.float() / (3 - 1)
        label_scaled = label_scaled.unsqueeze(1)
        inputs = torch.cat([latent_code_expanded, label_scaled, points], dim=1)
        predicted_sdf = model.decoder(inputs)

        loss = criterion(predicted_sdf, sdf)
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.6f}")

    visualize_shape_with_latent(model, latent_code, label, shape_count, grid_size, grid_range, device)


if __name__ == "__main__":
    latent_dim = 64
    hidden_dim = 512
    num_layers = 16
    num_embeddings = 627
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    test_dataset = DeepSDFDataset2D('../data_shapeNet', split='test')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for i in range(1):
        infer_and_visualize_shape(trained_model, test_loader, 100, grid_size=500, grid_range=(-448, 448), device=device,
                                  num_iterations=20000, lr=1e-2)
