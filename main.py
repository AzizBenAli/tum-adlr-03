from scripts.data_manipulation.data_preprocessing import MeshProcessor
from scripts.data_manipulation.data_loader import DeepSDFDataset2D
from scripts.models.decoder import DeepSDFModel
from scripts.training.train import train_model
from scripts.testing.test import infer_and_visualize_shape
from torch.utils.data import DataLoader
from helper import *

if __name__ == "__main__":
    config = load_config("configs/config.yaml")
    mode = "multi_class"
    mode_config = config[mode]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(mode, exist_ok=True)

    parent_directory = mode_config["parent_directory"]
    output_base_dir = mode_config["output_base_dir"]
    data_folder = mode_config["data_folder"]
    trained_models_dir = mode_config["trained_models_dir"]
    plots_dir = mode_config["plots_dir"]
    os.makedirs(output_base_dir, exist_ok=True)

    shapes = mode_config["shapes"]
    max_objects_per_type = mode_config["max_objects_per_type"]

    latent_dim = mode_config["latent_dim"]
    hidden_dim = mode_config["hidden_dim"]
    num_layers = mode_config["num_layers"]
    batch_size_train = mode_config["batch_size_train"]
    batch_size_val = mode_config["batch_size_val"]
    lr_decoder = mode_config["lr_decoder"]
    lr_latent = mode_config["lr_latent"]
    latent_reg_weight = mode_config["latent_reg_weight"]
    num_epochs = mode_config["num_epochs"]

    grid_size = mode_config["grid_size"]
    grid_range = tuple(mode_config["grid_range"])
    num_iterations = mode_config["num_iterations"]
    lr_visualization = mode_config["lr_visualization"]

    processor = MeshProcessor(parent_directory, output_base_dir, max_objects_per_type)
    for shape in shapes:
        processor.process_folder(shape)

    train_dataset = DeepSDFDataset2D(data_folder, split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_dataset = DeepSDFDataset2D(data_folder, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False)

    num_embeddings = len(set(train_dataset.shape_indices.tolist()))
    model = DeepSDFModel(latent_dim, hidden_dim, num_layers, num_embeddings)
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam([
        {"params": model.latent_codes.parameters(), "lr": lr_latent},
        {"params": model.decoder.parameters(), "lr": lr_decoder},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=True
    )

    trained_model = train_model(
        model, train_loader, criterion, optimizer, scheduler,
        num_epochs=num_epochs, latent_reg_weight=latent_reg_weight
    )

    os.makedirs(trained_models_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(trained_models_dir, "deepsdf_model.pth"))
    latent_codes = trained_model.get_all_latent_codes()
    np.save(os.path.join(trained_models_dir, "latent_codes.npy"), latent_codes.cpu().numpy())

    os.makedirs(plots_dir, exist_ok=True)
    for i in range(5):
        infer_and_visualize_shape(
            trained_model, test_dataset, test_loader, i, plots_dir,
            grid_size=grid_size, grid_range=grid_range, device=device,
            num_iterations=num_iterations, lr=lr_visualization
        )

    print("Training, testing and visualization completed.")
