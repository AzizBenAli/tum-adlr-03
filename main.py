import argparse
import os
import torch
import numpy as np
from scripts.data_manipulation.data_preprocessing import MeshProcessor
from scripts.models.decoder import DeepSDFModel
from scripts.training.train import train_model
from scripts.utils.config_utils import load_config
from scripts.data_manipulation.data_loader import DeepSDFDataset2D
from torch.utils.data import DataLoader
from scripts.helpers.visualization_helpers import infer_and_visualize_shape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepSDF Training and Testing Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["multi_class", "single_class"],
        help="Specify the mode for training",
    )
    args = parser.parse_args()

    config_settings = load_config("configs/settings.yaml")
    config_hyperparameters = load_config("configs/hyperparameters.yaml")
    mode = args.mode
    mode_config_settings = config_settings[mode]
    mode_config_hyperparameters = config_hyperparameters[mode]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(mode, exist_ok=True)

    parent_directory = mode_config_settings["parent_directory"]
    output_base_dir = mode_config_settings["output_base_dir"]
    data_folder = mode_config_settings["data_folder"]
    trained_models_dir = mode_config_settings["trained_models_dir"]
    plots_dir = mode_config_settings["plots_dir"]
    os.makedirs(output_base_dir, exist_ok=True)

    shapes = mode_config_settings["shapes"]
    max_objects_per_type = mode_config_settings["max_objects_per_type"]

    latent_dim = mode_config_hyperparameters["latent_dim"]
    hidden_dim = mode_config_hyperparameters["hidden_dim"]
    num_layers = mode_config_hyperparameters["num_layers"]
    batch_size_train = mode_config_hyperparameters["batch_size_train"]
    batch_size_val = mode_config_hyperparameters["batch_size_val"]
    lr_decoder = mode_config_hyperparameters["lr_decoder"]
    lr_latent = mode_config_hyperparameters["lr_latent"]
    latent_reg_weight = mode_config_hyperparameters["latent_reg_weight"]
    num_epochs = mode_config_hyperparameters["num_epochs"]

    grid_size = mode_config_settings["grid_size"]
    grid_range = tuple(mode_config_settings["grid_range"])
    num_iterations = mode_config_settings["num_iterations"]
    lr_visualization = mode_config_settings["lr_visualization"]

    processor = MeshProcessor(parent_directory, output_base_dir, max_objects_per_type)
    for shape in shapes:
        processor.process_folder(shape)

    train_dataset = DeepSDFDataset2D(data_folder, split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_dataset = DeepSDFDataset2D(data_folder, split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False)

    num_embeddings = len(set(train_dataset.shape_indices.tolist()))
    model = DeepSDFModel(latent_dim, hidden_dim, num_layers, num_embeddings)
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        [
            {"params": model.latent_codes.parameters(), "lr": lr_latent},
            {"params": model.decoder.parameters(), "lr": lr_decoder},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    trained_model = train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=num_epochs,
        latent_reg_weight=latent_reg_weight,
    )

    os.makedirs(trained_models_dir, exist_ok=True)
    torch.save(
        trained_model.state_dict(),
        os.path.join(trained_models_dir, "deepsdf_model.pth"),
    )
    latent_codes = trained_model.get_all_latent_codes()
    np.save(
        os.path.join(trained_models_dir, "latent_codes.npy"), latent_codes.cpu().numpy()
    )

    os.makedirs(plots_dir, exist_ok=True)
    for i in range(5):
        infer_and_visualize_shape(
            trained_model,
            test_dataset,
            test_loader,
            shape_idx=i,
            plots_dir=plots_dir,
            grid_size=grid_size,
            grid_range=grid_range,
            device=device,
            num_iterations=num_iterations,
            lr=lr_visualization,
            sampling_method="all",
        )

    print("Training, evaluation and visualization completed.")
