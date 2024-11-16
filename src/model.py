# deepsdf_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSDFDecoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, num_layers=8):
        super(DeepSDFDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input layer: concatenated latent code and 2D point
        self.fc_input = nn.Linear(latent_dim + 2, hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])

        # Output layer
        self.fc_output = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, latent_code, points):
        """
        Args:
            latent_code: Tensor of shape (batch_size, latent_dim)
            points: Tensor of shape (batch_size, 2)
        Returns:
            sdf: Tensor of shape (batch_size, 1)
        """
        # Concatenate latent code and points
        x = torch.cat([latent_code, points], dim=1)  # Shape: (batch_size, latent_dim + 3)
        x = F.relu(self.fc_input(x))

        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Output SDF value
        sdf = self.fc_output(x)
        return sdf


class DeepSDFModel(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, num_layers=8, num_embeddings=100):
        super(DeepSDFModel, self).__init__()
        self.latent_dim = latent_dim
        self.decoder = DeepSDFDecoder(latent_dim, hidden_dim, num_layers)

        # Initialize latent codes as learnable parameters
        self.latent_codes = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=latent_dim)

    def forward(self, shape_indices, points):
        latent_code = self.latent_codes(shape_indices)
        sdf = self.decoder(latent_code, points)
        return sdf
    def get_latent_codes(self):
        return self.latent_codes.weight.data.cpu().numpy()
