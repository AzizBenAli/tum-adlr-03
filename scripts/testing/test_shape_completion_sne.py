import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ------------------------------------------------------------------------
#  Your imports from your existing code
# ------------------------------------------------------------------------
from helper import visualize_shape_with_latent_shape_completion
from scripts.data_manipulation.data_loader import DeepSDFDataset2D
from scripts.models.decoder import DeepSDFModel

# ------------------------------------------------------------------------
#  Reproducibility settings
# ------------------------------------------------------------------------
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------------
#  1. Function to plot t-SNE with random init
# ------------------------------------------------------------------------
def plot_tsne_with_random_init(model, random_latent_code, device="cpu"):
    """
    Plots a t-SNE of the learned latent space together with the random
    initialization vector used for shape completion.

    Args:
        model (DeepSDFModel): The trained DeepSDF model.
        random_latent_code (torch.Tensor): The random latent code tensor
            (shape: [1, latent_dim]) used for shape completion (before optimization).
        device (str): "cpu" or "cuda".
    """
    # Ensure we're on CPU to avoid issues with sklearn
    random_latent_code_cpu = random_latent_code.detach().cpu()

    # ---------------------------------------------------------------------
    # 1) Extract the learned latent embeddings from your model
    #    Adjust if your embeddings are stored elsewhere.
    # ---------------------------------------------------------------------
    learned_latent_embeddings = model.latent_codes.weight.detach().cpu()
    # shape -> [num_shapes, latent_dim]

    # ---------------------------------------------------------------------
    # 2) Combine with the random initialization code
    # ---------------------------------------------------------------------
    all_embeddings = torch.cat([learned_latent_embeddings, random_latent_code_cpu], dim=0)
    # shape -> [num_shapes + 1, latent_dim]

    # ---------------------------------------------------------------------
    # 3) Run t-SNE on all embeddings
    # ---------------------------------------------------------------------
    all_embeddings_np = all_embeddings.numpy()
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="random",
        random_state=42
    )
    all_embeddings_2d = tsne.fit_transform(all_embeddings_np)  # shape -> [N, 2]

    # Separate learned codes and the random code in 2D
    learned_2d = all_embeddings_2d[:-1]  # All but the last
    random_2d = all_embeddings_2d[-1]    # The last row is the random code

    # ---------------------------------------------------------------------
    # 4) Plot the t-SNE
    # ---------------------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.scatter(learned_2d[:, 0], learned_2d[:, 1],
                alpha=0.7, label="Learned Latent Codes", color="blue")
    plt.scatter(random_2d[0], random_2d[1],
                s=120, color="red", label="Random Initialization")
    plt.title("t-SNE of Latent Space + Random Initialization")
    plt.legend()
    plt.show()

# ------------------------------------------------------------------------
# 2. Inference and visualization function
# ------------------------------------------------------------------------
def infer_and_visualize_shape(
    model,
    test_dataset,
    test_data_loader,
    shape_idx,
    plots_dir,
    grid_size=224,
    grid_range=(-10, 10),
    device="cpu",
    num_iterations=500,
    lr=1e-2,
    # Pass in an initial_latent_code so we know where we started
    initial_latent_code=None
):
    """
    Given a shape index, retrieves partial observations (points+sdf),
    optimizes a latent code to fit those partial observations,
    and visualizes the shape.
    """
    model.eval()

    # ---------------------------------------------------------------------
    #  Collect all points/SDF for the target shape index
    # ---------------------------------------------------------------------
    full_points = []
    full_sdf = []
    shape_count = 0

    for batch in test_data_loader:
        batch_shape_idx = batch["shape_idx"]
        if batch_shape_idx[0].item() == shape_idx:
            full_points.append(batch["point"].to(device).float())
            full_sdf.append(batch["sdf"].to(device).float())
            shape_count = batch["shape_count"][0]

    full_points = torch.cat(full_points, dim=0)
    full_sdf = torch.cat(full_sdf, dim=0)
    print("Full points shape:", full_points.shape)

    # ---------------------------------------------------------------------
    #  Subsample partial observations
    # ---------------------------------------------------------------------
    num_samples = full_points.shape[0]
    half_samples = num_samples // 30
    indices = torch.randperm(num_samples)[:half_samples]
    partial_points = full_points[indices]
    partial_sdf = full_sdf[indices]

    # ---------------------------------------------------------------------
    #  Set up the latent code to optimize
    #     - If an initial code is provided, use that. Otherwise create new.
    # ---------------------------------------------------------------------
    if initial_latent_code is not None:
        latent_code = initial_latent_code.clone().detach().to(device)
    else:
        latent_code = torch.normal(
            mean=0.0,
            std=0.01,
            size=(1, model.latent_dim),
        ).to(device)

    latent_code.requires_grad = True

    # ---------------------------------------------------------------------
    #  Optimize the latent code
    # ---------------------------------------------------------------------
    optimizer = torch.optim.Adam([latent_code], lr=lr)
    criterion = torch.nn.MSELoss()

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        latent_code_expanded = latent_code.repeat(partial_points.shape[0], 1)
        predicted_sdf = model.decoder(latent_code_expanded, partial_points)
        loss = criterion(predicted_sdf, partial_sdf)
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.6f}")

    # ---------------------------------------------------------------------
    #  Visualize the reconstructed/estimated shape
    # ---------------------------------------------------------------------
    visualize_shape_with_latent_shape_completion(
        model,
        test_dataset,
        latent_code,
        shape_count,
        plots_dir,
        grid_size,
        grid_range,
        device,
        partial_points
    )

    # Return the final (optimized) latent code
    return latent_code

# ------------------------------------------------------------------------
# 3. Main script logic
# ------------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 64
    hidden_dim = 512
    num_layers = 32
    num_embeddings = 314

    # Load trained model
    trained_model = DeepSDFModel(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_embeddings=num_embeddings
    )
    state_dict_path = "../../multi_class/trained_models/deepsdf_model.pth"
    state_dict = torch.load(state_dict_path, map_location=device)
    trained_model.load_state_dict(state_dict)
    trained_model.to(device)
    trained_model.eval()

    # Load dataset & create loader
    test_dataset = DeepSDFDataset2D("../../multi_class/data", split="test")
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

    # Where to store final shape visualization
    plots_dir = "../../multi_class/plots"

    # ---------------------------------------------------------------------
    # 3a) Create a random latent code for shape completion
    #     We will pass this *before* optimization into t-SNE
    # ---------------------------------------------------------------------
    random_init_code = torch.normal(
        mean=0.0,
        std=0.01,
        size=(1, trained_model.latent_dim)
    ).to(device)

    # ---------------------------------------------------------------------
    # 3b) Plot the t-SNE with the random initialization shown
    # ---------------------------------------------------------------------
    plot_tsne_with_random_init(
        model=trained_model,
        random_latent_code=random_init_code,
        device=device
    )

    # ---------------------------------------------------------------------
    # 3c) Run shape completion using that same random_init_code
    # ---------------------------------------------------------------------
    final_code = infer_and_visualize_shape(
        model=trained_model,
        test_dataset=test_dataset,
        test_data_loader=test_loader,
        shape_idx=45,              # Example shape
        plots_dir=plots_dir,
        grid_size=500,
        grid_range=(-448, 448),
        device=device,
        num_iterations=2600,
        lr=1e-1,
        initial_latent_code=random_init_code
    )

    # If desired, you could do another t-SNE after optimization to see
    # where the final code ended up. Just call plot_tsne_with_random_init()
    # again, but pass in final_code instead of random_init_code.
    #
    # e.g.,
    plot_tsne_with_random_init(
         model=trained_model,
         random_latent_code=final_code,
         device=device
     )
