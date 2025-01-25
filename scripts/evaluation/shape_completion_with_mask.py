import torch
from scripts.data_manipulation.data_loader import DeepSDFDataset2D
from torch.utils.data import DataLoader
from scripts.models.decoder import DeepSDFModel
from scripts.helpers.visualization_helpers import infer_and_visualize_shape

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    latent_dim = 64
    hidden_dim = 512
    num_layers = 32
    num_embeddings = 314

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trained_model = DeepSDFModel(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_embeddings=num_embeddings,
    )
    state_dict_path = "../../multi_class/trained_models/deepsdf_model.pth"
    state_dict = torch.load(state_dict_path, map_location=torch.device(device))
    trained_model.load_state_dict(state_dict)
    trained_model.to(device)
    trained_model.eval()

    test_dataset = DeepSDFDataset2D("../../multi_class/data", split="test")
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

    plots_dir = "../../multi_class/visualization"

    infer_and_visualize_shape(
        trained_model,
        test_dataset,
        test_loader,
        shape_idx=19,
        plots_dir=plots_dir,
        grid_size=500,
        grid_range=(-448, 448),
        device=device,
        num_iterations=100,
        lr=1e-1,
        sampling_method="mask",
    )
