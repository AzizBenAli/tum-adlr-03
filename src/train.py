# train_deepsdf.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.output.Data_loader import DeepSDFDataset2D
from model import DeepSDFModel
import os
import numpy as np


def train_model(model, dataloader, criterion, optimizer, num_epochs=50, device='cpu'):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in dataloader:
            shape_indices = batch['shape_idx'].to(device)
            points = batch['point'].to(device)
            sdf = batch['sdf'].to(device)

            # Forward pass
            optimizer.zero_grad()
            predicted_sdf = model(shape_indices, points)
            if epoch == 99:
                print(predicted_sdf)
            # Compute loss
            loss = criterion(predicted_sdf, sdf)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()


        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model


if __name__ == "__main__":
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load training dataset
    data_folder = '/Users/yahyaabdelhamed/Documents/tum-adlr-03/src/output_mnist'
    train_dataset = DeepSDFDataset2D(data_folder, split='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Determine the total number of unique shapes
    num_embeddings = len(set(train_dataset.shape_indices.tolist()))
    # Initialize model, loss, and optimizer
    latent_dim = 78
    model = DeepSDFModel(latent_dim=latent_dim, num_embeddings=num_embeddings).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=100, device=device)

    # Save trained model and latent codes
    os.makedirs('trained_models', exist_ok=True)
    torch.save(trained_model.state_dict(), 'trained_models/deepsdf_model.pth')
    latent_codes = trained_model.get_latent_codes()
    np.save('trained_models/latent_codes.npy', latent_codes)

    print("Model and latent codes saved.")

