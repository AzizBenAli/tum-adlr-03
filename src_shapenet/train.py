import sys
sys.path.append('/content/tum-adlr-03-shapenet')
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import DeepSDFDataset2D
from model import DeepSDFModel
from utils import *
import os
import numpy as np

def validate_model(model, dataloader, criterion, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0
    total_latent_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            shape_indices = batch['shape_idx'].to(device).long()
            points = batch['point'].to(device).float()
            sdf = batch['sdf'].to(device).float()

            predicted_sdf = model(shape_indices, points)
            latent_codes = model.get_latent_codes(shape_indices)
            latent_loss = torch.mean(latent_codes.pow(2))
            sdf_loss = criterion(predicted_sdf, sdf)

            total_val_loss += sdf_loss.item()
            total_latent_loss += latent_loss.item()

    avg_val_sdf_loss = total_val_loss / len(dataloader)
    avg_val_latent_loss = total_latent_loss / len(dataloader)

    return avg_val_sdf_loss, avg_val_latent_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device='cpu', latent_reg_weight=1):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_sdf_loss = 0.0
        total_latent_loss = 0.0

        for batch in train_loader:
            shape_indices = batch['shape_idx'].to(device).long()
            points = batch['point'].to(device).float()
            sdf = batch['sdf'].to(device).float()
            optimizer.zero_grad()
            predicted_sdf = model(shape_indices, points)

            latent_codes = model.get_latent_codes(shape_indices)
            latent_loss = torch.mean((latent_codes**2))
            sdf_loss = criterion(predicted_sdf, sdf)
            total_loss_batch = sdf_loss + latent_reg_weight * latent_loss
            total_sdf_loss += sdf_loss.item()
            total_latent_loss += latent_loss.item()

            total_loss_batch.backward()
            optimizer.step()

        avg_sdf_loss = total_sdf_loss / len(train_loader)
        avg_latent_loss = total_latent_loss / len(train_loader)
        if scheduler:
            scheduler.step(avg_sdf_loss)
        # Validate the model
        avg_val_sdf_loss, avg_val_latent_loss = validate_model(model, val_loader, criterion, device)

        # Adjust the learning rate based on validation loss


        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train SDF Loss: {avg_sdf_loss:.6f}, Train Latent Loss: {avg_latent_loss:.6f}")
        #print(f"Val SDF Loss: {avg_val_sdf_loss:.6f}, Val Latent Loss: {avg_val_latent_loss:.6f}")
       # print("-" * 50)

    return model



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_folder = '../data_shapeNet'
    train_dataset = DeepSDFDataset2D(data_folder, split='train')
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_dataset = DeepSDFDataset2D(data_folder, split='test')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_embeddings = len(set(train_dataset.shape_indices.tolist()))
    print(f"Number of embeddings: {num_embeddings}")
    latent_dim = 64
    model = DeepSDFModel(latent_dim=latent_dim,hidden_dim=512,num_layers=16, num_embeddings=num_embeddings).to(device)
    criterion = nn.MSELoss()

    # Adjusted learning rates
    lr_decoder = 0.0001
    lr_latent = 0.001
    optimizer = torch.optim.Adam([
        {"params": model.latent_codes.parameters(), "lr": lr_latent},
        {"params": model.decoder.parameters(), "lr": lr_decoder}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=True
    )

    # Adjusted latent regularization weight
    latent_reg_weight = 0.0001

    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=100, device=device, latent_reg_weight=latent_reg_weight
    )

    os.makedirs('../trained_models', exist_ok=True)
    torch.save(trained_model.state_dict(), '../trained_models/deepsdf_model.pth')
    latent_codes = trained_model.get_all_latent_codes()
    np.save('../trained_models/latent_codes.npy', latent_codes.cpu().numpy())

    visualize_predictions(trained_model, val_dataset, shape_idx=0, device=device)
    visualize_latent_codes(model, val_loader, device=device)

    print("Model and latent codes saved.")

