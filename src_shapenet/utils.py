import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from PIL import Image

def plot_border_points_on_mask(border_points, image_array, output_directory, image_name):
    # Convert the image array to a 3D array (if it's not already), making it an RGB image
    plot_image = np.stack([image_array] * 3, axis=-1)  # Stack grayscale to create RGB channels

    # Annotate the image with border points
    for point in border_points:
        x, y, sdf, location = point
        if location == "inside":
            plot_image[x, y] = [255, 0, 0]  # Red color for inside border
        else:
            plot_image[x, y] = [0, 0, 255]  # Blue color for outside border

    # Create the "annotated_images" directory if it doesn't exist
    annotated_images_dir = os.path.join(output_directory, "annotated_images")
    os.makedirs(annotated_images_dir, exist_ok=True)

    # Convert the numpy array back to a PIL image
    annotated_image = Image.fromarray(plot_image.astype(np.uint8))

    # Save the image to the "annotated_images" folder
    annotated_image_path = os.path.join(annotated_images_dir, f"{image_name}_annotated.png")
    annotated_image.save(annotated_image_path)

    print(f"Image saved to: {annotated_image_path}")

def visualize_predictions(model, dataset, shape_idx, device='cpu'):
    model.eval()
    with torch.no_grad():
        # Filter points and corresponding SDF values that belong to the shape_idx
        points = []
        true_sdf = []

        # Iterate through the entire dataset and collect points and SDF values for the given shape_idx
        for data in dataset:
            if data['shape_idx'] == shape_idx:
                points.append(data['point'].to(device))  # Add points
                true_sdf.append(data['sdf'].to(device))  # Add corresponding SDF values

        points = torch.stack(points, dim=0)
        true_sdf = torch.stack(true_sdf, dim=0)

        # Create a tensor of repeated shape indices corresponding to all the points
        shape_idx_tensor = torch.tensor([shape_idx] * len(points), device=device)

        # Predict the SDF for the gathered points
        predicted_sdf = model(shape_idx_tensor, points)

        # Plot a scatter plot of predicted vs true SDF values
        plt.figure(figsize=(8, 8))
        plt.scatter(predicted_sdf.cpu(), true_sdf.cpu(), c='blue', label='Predicted vs True', alpha=0.5)

        # Identity line: Where predicted SDF = true SDF
        min_sdf = min(predicted_sdf.min(), true_sdf.min())
        max_sdf = max(predicted_sdf.max(), true_sdf.max())
        plt.plot([min_sdf, max_sdf], [min_sdf, max_sdf], 'k--', label="Identity Line (x = y)")

        os.makedirs('../plots', exist_ok=True)

        # Labels and title
        plt.xlabel("Predicted SDF")
        plt.ylabel("True SDF")
        plt.title(f"Predicted vs True SDF for Shape {shape_idx}")
        plt.legend()
        plt.savefig("../plots/actual_vs_predicted_sdf.png")
        plt.grid(True)

        plt.show()


def visualize_latent_codes(model, dataloader, device='cpu'):
    latent_codes = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            shape_indices = batch['shape_idx'].to(device)
            latent_codes_batch = model.get_latent_codes(shape_indices)
            latent_codes.append(latent_codes_batch.cpu().numpy())
            labels.append(batch['label_idx'].cpu().numpy())

    latent_codes = np.concatenate(latent_codes, axis=0)
    labels = np.concatenate(labels, axis=0)


    tsne = TSNE(n_components=2, random_state=42)
    latent_codes_2d = tsne.fit_transform(latent_codes)

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(latent_codes_2d[:, 0], latent_codes_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)

    os.makedirs('../plots', exist_ok=True)

    plt.legend(handles=scatter.legend_elements()[0], labels=[f"Class {i}" for i in range(4)])
    plt.title('Latent Space Visualization (t-SNE)')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.colorbar(scatter, label='Class Label')
    plt.savefig("../plots/latent_codes.png")
    plt.show()
