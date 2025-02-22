import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_shape_with_latent(
    model,
    test_dataset,
    latent_code,
    shape_count,
    plots_dir,
    grid_size=224,
    grid_range=(-10, 10),
    device="cpu",
):
    grid_size = int(grid_size)
    model.eval()
    with torch.no_grad():
        x = np.linspace(grid_range[0], grid_range[1], grid_size)
        y = np.linspace(grid_range[0], grid_range[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        points = torch.tensor(points, dtype=torch.float32).to(device)

        latent_code_expanded = latent_code.repeat(points.shape[0], 1)

        sdf_values = model.decoder(latent_code_expanded, points)
        sdf_values = sdf_values.cpu().numpy().reshape((grid_size, grid_size))

        sdf_values = np.rot90(sdf_values, k=1)

        image = np.zeros((grid_size, grid_size), dtype=np.uint8)
        image[sdf_values < 0] = 255

        shape_count = int(shape_count[0])
        annotated_image_path = f"shape{shape_count}_annotated.png"
        print(annotated_image_path)
        image_dir = os.path.join(test_dataset.root_folder, "data", "annotated_images")
        image_path = os.path.join(image_dir, annotated_image_path)

        annotated_image = plt.imread(image_path)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(
            image,
            extent=(grid_range[0], grid_range[1], grid_range[0], grid_range[1]),
            origin="lower",
            cmap="gray",
        )
        axes[0].set_title("Reconstructed Shape")
        axes[0].axis("off")
        axes[0].set_aspect("equal")

        axes[1].imshow(annotated_image, cmap="gray")
        axes[1].set_title("Annotated Shape")
        axes[1].axis("off")
        axes[1].set_aspect("equal")

        plt.tight_layout()
        plt.savefig(f"{plots_dir}/reconstructed_image_{shape_count}")
        plt.show()


def visualize_shape_with_latent_shape_completion(
    model,
    test_dataset,
    latent_code,
    shape_count,
    plots_dir,
    grid_size=224,
    grid_range=(-10, 10),
    device="cpu",
    optimization_points=None,
    sampling_method="random",
):
    grid_size = int(grid_size)
    model.eval()

    with torch.no_grad():
        x = np.linspace(grid_range[0], grid_range[1], grid_size)
        y = np.linspace(grid_range[0], grid_range[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        points = torch.tensor(points, dtype=torch.float32).to(device)

        latent_code_expanded = latent_code.repeat(points.shape[0], 1)
        sdf_values = model.decoder(latent_code_expanded, points)
        sdf_values = sdf_values.cpu().numpy().reshape((grid_size, grid_size))

        sdf_values = np.rot90(sdf_values, k=1)

        image = np.zeros((grid_size, grid_size), dtype=np.uint8)
        image[sdf_values < 0] = 255

        shape_count = int(shape_count[0])
        annotated_image_path = f"shape{shape_count}_annotated.png"
        image_dir = os.path.join(test_dataset.root_folder, "data", "annotated_images")
        image_path = os.path.join(image_dir, annotated_image_path)
        annotated_image = plt.imread(image_path)

        fig, axes = plt.subplots(1, 3, figsize=(30, 10))

        axes[0].imshow(
            image,
            origin="lower",
            cmap="gray",
        )
        axes[0].set_title("Reconstructed Shape")
        axes[0].axis("off")
        axes[0].set_aspect("equal")

        if optimization_points is not None:
            optimized_points = optimization_points.cpu().numpy()

            shift = -grid_range[0]
            scaled_x = (optimized_points[:, 0] + shift) * (
                grid_size / (grid_range[1] - grid_range[0])
            )
            scaled_y = (optimized_points[:, 1] + shift) * (
                grid_size / (grid_range[1] - grid_range[0])
            )

            scaled_x = np.clip(scaled_x, 0, grid_size - 1)
            scaled_y = np.clip(scaled_y, 0, grid_size - 1)

            axes[2].scatter(
                scaled_y, -scaled_x, color="green", s=10, label="Optimization Points"
            )
            axes[2].legend()

        axes[1].imshow(annotated_image, cmap="gray")
        axes[1].set_title("Annotated Shape")
        axes[1].axis("off")
        axes[1].set_aspect("equal")

        plt.tight_layout()
        plt.savefig(
            f"{plots_dir}/reconstructed_image_{shape_count}_shape_completion_{sampling_method}.png"
        )
        plt.show()


def visualize_latent_space(model, device="cpu"):
    latent_codes = model.latent_codes.weight.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    latent_codes_2d = tsne.fit_transform(latent_codes)

    plt.figure(figsize=(10, 8))
    plt.scatter(latent_codes_2d[:, 0], latent_codes_2d[:, 1], alpha=0.7)
    plt.title("Latent Space Visualization using t-SNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()


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
    sampling_method="random",
    max_points=200,
    random_sampling_fraction=30,
):
    if sampling_method not in {"random", "mask", "all"}:
        raise ValueError(
            f"Invalid sampling method: {sampling_method}. Please use 'random' or 'mask'."
        )

    model.eval()

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

    if sampling_method == "mask":
        x_coords = full_points[:, 0]
        y_coords = full_points[:, 1]
        x_middle = (x_coords.min() + x_coords.max()) / 2
        y_middle = (y_coords.min() + y_coords.max()) / 2

        intersection_mask = (y_coords < y_middle) & (x_coords < x_middle)
        partial_points = full_points[intersection_mask]
        partial_sdf = full_sdf[intersection_mask]

        print(
            f"Sampling using 'mask' method: Selected {partial_points.shape[0]} "
            f"points below the middle of the x-axis and y-axis."
        )

        num_selected_points = min(max_points, partial_points.shape[0])
        selected_indices = torch.randperm(partial_points.shape[0])[:num_selected_points]
        partial_points = partial_points[selected_indices]
        partial_sdf = partial_sdf[selected_indices]

    elif sampling_method == "random":
        num_samples = full_points.shape[0]
        half_samples = num_samples // random_sampling_fraction
        indices = torch.randperm(num_samples)[:half_samples]
        partial_points = full_points[indices]
        partial_sdf = full_sdf[indices]

        print(
            f"Sampling using 'random' method: Selected {partial_points.shape[0]} random points."
        )

        num_selected_points = partial_points.shape[0]
        selected_indices = torch.randperm(partial_points.shape[0])[:num_selected_points]
        partial_points = partial_points[selected_indices]
        partial_sdf = partial_sdf[selected_indices]

    elif sampling_method == "all":
        partial_points = full_points
        partial_sdf = full_sdf
        print("Sampling using 'all' method: Using all points.")

        num_selected_points = partial_points.shape[0]
        selected_indices = torch.randperm(partial_points.shape[0])[:num_selected_points]
        partial_points = partial_points[selected_indices]
        partial_sdf = partial_sdf[selected_indices]

    latent_mean = 0.0
    latent_std = 0.01
    latent_code = torch.normal(
        mean=latent_mean,
        std=latent_std,
        size=(1, model.latent_dim),
        device=device,
        requires_grad=True,
    )

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
            print(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item():.6f}")

    visualize_shape_with_latent_shape_completion(
        model,
        test_dataset,
        latent_code,
        shape_count,
        plots_dir,
        grid_size,
        grid_range,
        device,
        partial_points,
        sampling_method,
    )


def visualize_reconstructed_shape(
    model,
    latent_code,
    plots_dir,
    grid_size=224,
    grid_range=(-10, 10),
    device="cpu",
    save_path="",
):
    grid_size = int(grid_size)
    model.eval()

    with torch.no_grad():
        x = np.linspace(grid_range[0], grid_range[1], grid_size)
        y = np.linspace(grid_range[0], grid_range[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        points = torch.tensor(points, dtype=torch.float32).to(device)

        latent_code_expanded = latent_code.repeat(points.shape[0], 1)
        sdf_values = model.decoder(latent_code_expanded, points)
        sdf_values = sdf_values.cpu().numpy().reshape((grid_size, grid_size))

        sdf_values = np.rot90(sdf_values, k=1)

        image = np.zeros((grid_size, grid_size), dtype=np.uint8)
        image[sdf_values < 0] = 255

        fig, axes = plt.subplots(1, 1, figsize=(30, 10))

        axes.imshow(
            image,
            extent=(grid_range[0], grid_range[1], grid_range[0], grid_range[1]),
            origin="lower",
            cmap="gray",
        )
        axes.set_title("Reconstructed Shape")
        axes.axis("off")
        axes.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def plot_tsne_with_random_init(model, random_latent_code, save_path=None):
    random_latent_code_cpu = random_latent_code.detach().cpu()

    learned_latent_embeddings = model.latent_codes.weight.detach().cpu()
    all_embeddings = torch.cat(
        [learned_latent_embeddings, random_latent_code_cpu], dim=0
    )
    all_embeddings_np = all_embeddings.numpy()
    tsne = PCA(n_components=2, random_state=42)
    all_embeddings_2d = tsne.fit_transform(all_embeddings_np)

    learned_2d = all_embeddings_2d[:-1]
    random_2d = all_embeddings_2d[-1]

    plt.figure(figsize=(8, 8))
    plt.scatter(
        learned_2d[:, 0],
        learned_2d[:, 1],
        alpha=0.7,
        label="Learned Latent Codes",
        color="blue",
    )
    plt.scatter(
        random_2d[0], random_2d[1], s=120, color="red", label="Random Initialization"
    )
    plt.title("t-SNE of Latent Space + Random Initialization")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
