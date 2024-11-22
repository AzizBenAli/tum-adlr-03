import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSDFDecoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, num_layers=8, use_skip_connections=True):
        super(DeepSDFDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_skip_connections = use_skip_connections

        # Input layer
        self.fc_input = nn.Linear(latent_dim + 2, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            if self.use_skip_connections and i == (num_layers // 2) - 1:
                # At the middle layer, concatenate residual
                self.hidden_layers.append(nn.Linear(hidden_dim + latent_dim + 2, hidden_dim))
            else:
                self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.fc_output = nn.Linear(hidden_dim, 1)

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, latent_code, points):
        """
        Args:
            latent_code: Tensor of shape (batch_size, latent_dim)
            points: Tensor of shape (batch_size, 2)
        Returns:
            sdf: Tensor of shape (batch_size, 1)
        """
        # Concatenate latent code and points
        x = torch.cat([latent_code, points], dim=1)
        residual = x if self.use_skip_connections else None

        # Input layer
        x = self.fc_input(x)
        x = self.bn_input(x)
        x = self.activation(x)

        # Pass through hidden layers
        for i, layer in enumerate(self.hidden_layers):
            if self.use_skip_connections and i == (self.num_layers // 2) - 1:
                x = torch.cat([x, residual], dim=1)
            x = layer(x)
            x = self.bn_layers[i](x)
            x = self.activation(x)

        # Output layer
        sdf = self.fc_output(x)
        return sdf


class DeepSDFModel(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, num_layers=8, num_embeddings=69, use_skip_connections=True):
        super(DeepSDFModel, self).__init__()
        self.latent_dim = latent_dim
        self.decoder = DeepSDFDecoder(latent_dim, hidden_dim, num_layers, use_skip_connections)

        # Embedding for latent codes
        self.latent_codes = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0.0, std=0.01)

    def forward(self, shape_indices, points):
        latent_code = self.latent_codes(shape_indices)
        sdf = self.decoder(latent_code, points)
        return sdf

    def get_all_latent_codes(self):
        return self.latent_codes.weight.data

    def get_latent_codes(self, shape_indices):
        return self.latent_codes(shape_indices)
