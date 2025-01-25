import numpy as np
import torch
from scripts.data_manipulation.data_loader import DeepSDFDataset2D
from torch.utils.data import DataLoader


def interpolate_latent_codes(z1, z2, num_steps=10):
    interpolated_latents = []
    for t in np.linspace(0, 1, num_steps):
        z_t = (1 - t) * z1 + t * z2
        interpolated_latents.append(z_t)
    return interpolated_latents


def slerp(z1, z2, num_steps=10):
    z1_norm = z1 / torch.norm(z1)
    z2_norm = z2 / torch.norm(z2)
    omega = torch.acos(
        torch.clamp(
            torch.dot(z1_norm.squeeze(), z2_norm.squeeze()), -1.0 + 1e-7, 1.0 - 1e-7
        )
    )
    sin_omega = torch.sin(omega)
    interpolated_latents = []
    for t in np.linspace(0, 1, num_steps):
        if sin_omega == 0:
            z_t = (1 - t) * z1 + t * z2
        else:
            z_t = (torch.sin((1 - t) * omega) / sin_omega) * z1 + (
                torch.sin(t * omega) / sin_omega
            ) * z2
        interpolated_latents.append(z_t)
    return interpolated_latents


def search_train_dataset(shape_idx):
    train_dataset = DeepSDFDataset2D("../../multi_class/data", split="train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    for batch in train_loader:
        batch_shape_idx = batch["shape_idx"]
        if batch_shape_idx[0].item() == shape_idx:
            return batch["shape_count"]
    else:
        print(f"Shape index {shape_idx} not found in test data.")
        return
