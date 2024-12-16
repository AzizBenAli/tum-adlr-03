import torch
import torch.nn as nn

class DeepSDFDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=8, use_skip_connections=True):
        super(DeepSDFDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_skip_connections = use_skip_connections

        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)

        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            if self.use_skip_connections and i == (num_layers // 2) - 1:
                self.hidden_layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            else:
                self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        self.fc_output = nn.Linear(hidden_dim, 1)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):

        residual = x if self.use_skip_connections else None

        x = self.fc_input(x)
        x = self.bn_input(x)
        x = self.activation(x)

        for i, layer in enumerate(self.hidden_layers):
            if self.use_skip_connections and i == (self.num_layers // 2) - 1:
                x = torch.cat([x, residual], dim=1)
            x = layer(x)
            x = self.bn_layers[i](x)
            x = self.activation(x)

        sdf = self.fc_output(x)
        return sdf


class DeepSDFModel(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, num_layers=8, num_embeddings=100, num_labels=3, use_skip_connections=True):
        super(DeepSDFModel, self).__init__()
        self.latent_dim = latent_dim
        self.num_labels = num_labels

        input_dim = latent_dim  + 2
        self.decoder = DeepSDFDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_skip_connections=use_skip_connections
        )

        self.latent_codes = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0.0, std=0.01)

    def forward(self, shape_indices, label_indices, points):

        latent_code = self.latent_codes(shape_indices)

        label_scaled = label_indices.float() / (self.num_labels - 1)
        label_scaled = label_scaled.unsqueeze(1)

        input_vec = torch.cat([latent_code, label_scaled, points], dim=1)

        sdf = self.decoder(input_vec)
        return sdf

    def get_all_latent_codes(self):
        return self.latent_codes.weight.data

    def get_latent_codes(self, shape_indices):
        return self.latent_codes(shape_indices)
