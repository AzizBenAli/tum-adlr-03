import torch.nn as nn
from torch.utils.data import DataLoader
from src_shapenet.data_loader import DeepSDFDataset2D
from model import DeepSDFModel
from utils import *
import os
import numpy as np

def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=50, device='cpu', latent_reg_weight=1e-4):
    model.train()

    for epoch in range(num_epochs):
        total_sdf_loss = 0.0
        total_latent_loss = 0.0

        for batch in dataloader:
            shape_indices = batch['shape_idx'].to(device).long()
            points = batch['point'].to(device).float()
            sdf = batch['sdf'].to(device).float()

            optimizer.zero_grad()
            predicted_sdf = model(shape_indices, points)

            latent_codes = model.get_latent_codes(shape_indices)
            latent_loss = torch.mean(latent_codes.pow(2))
            sdf_loss = criterion(predicted_sdf, sdf)
            total_loss_batch = sdf_loss + latent_reg_weight * latent_loss
            total_sdf_loss += sdf_loss.item()
            total_latent_loss += latent_loss.item()

            total_loss_batch.backward()
            #for name, param in model.named_parameters():
                #if param.grad is not None:
                   # print(f"{name} grad norm: {param.grad.norm()}")
            optimizer.step()

        if scheduler:
            scheduler.step()

        avg_sdf_loss = total_sdf_loss / len(dataloader)
        avg_latent_loss = total_latent_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], SDF Loss: {avg_sdf_loss:.6f}, Latent Loss: {avg_latent_loss:.6f}")

    return model


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_folder = '../data_shapeNet'
    train_dataset = DeepSDFDataset2D(data_folder, split='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = DeepSDFDataset2D(data_folder, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    num_embeddings = len(set(train_dataset.shape_indices.tolist()))
    print(f"Number of embeddings: {num_embeddings}")
    latent_dim = 16
    model = DeepSDFModel(latent_dim=latent_dim, num_embeddings=num_embeddings).to(device)
    criterion = nn.MSELoss()

    # Adjusted learning rates
    lr_decoder = 1e-3
    lr_latent = 1e-3
    optimizer = torch.optim.Adam([
        {"params": model.latent_codes.parameters(), "lr": lr_latent},
        {"params": model.decoder.parameters(), "lr": lr_decoder}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Adjusted latent regularization weight
    latent_reg_weight = 1e-4

    trained_model = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=50, device=device, latent_reg_weight=latent_reg_weight)

    os.makedirs('../trained_models', exist_ok=True)
    torch.save(trained_model.state_dict(), '../trained_models/deepsdf_model.pth')
    latent_codes = trained_model.get_all_latent_codes()
    np.save('../trained_models/latent_codes.npy', latent_codes.cpu().numpy())

    visualize_predictions(trained_model, test_dataset, shape_idx=0, device=device)
    visualize_latent_codes(model, test_loader, device=device)

    print("Model and latent codes saved.")
