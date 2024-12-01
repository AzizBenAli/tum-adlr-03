import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def plot_border_points_on_mask(border_points, image_array, output_directory, image_name):
    plot_image = np.stack([image_array] * 3, axis=-1)

    for point in border_points:
        x, y, sdf, location, shape_count = point
        if location == "inside":
            plot_image[x, y] = [255, 0, 0]
        else:
            plot_image[x, y] = [0, 0, 255]

    annotated_images_dir = os.path.join(output_directory, "annotated_images")
    os.makedirs(annotated_images_dir, exist_ok=True)

    annotated_image = Image.fromarray(plot_image.astype(np.uint8))

    annotated_image_path = os.path.join(annotated_images_dir, f"{image_name}_annotated.png")
    annotated_image.save(annotated_image_path)

    print(f"Image saved to: {annotated_image_path}")

def visualize_predictions(model, dataset, shape_idx, device='cpu'):
    model.eval()
    with torch.no_grad():
        points = []
        true_sdf = []

        for data in dataset:
            if data['shape_idx'] == shape_idx:
                points.append(data['point'].to(device))
                true_sdf.append(data['sdf'].to(device))

        points = torch.stack(points, dim=0)
        true_sdf = torch.stack(true_sdf, dim=0)

        shape_idx_tensor = torch.tensor([shape_idx] * len(points), device=device)

        predicted_sdf = model(shape_idx_tensor, points)

        plt.figure(figsize=(8, 8))
        plt.scatter(predicted_sdf.cpu(), true_sdf.cpu(), c='blue', label='Predicted vs True', alpha=0.5)

        min_sdf = min(predicted_sdf.min(), true_sdf.min())
        max_sdf = max(predicted_sdf.max(), true_sdf.max())
        plt.plot([min_sdf, max_sdf], [min_sdf, max_sdf], 'k--', label="Identity Line (x = y)")

        os.makedirs('../plots', exist_ok=True)

        plt.xlabel("Predicted SDF")
        plt.ylabel("True SDF")
        plt.title(f"Predicted vs True SDF for Shape {shape_idx}")
        plt.legend()
        plt.savefig("../plots/actual_vs_predicted_sdf.png")
        plt.grid(True)

        plt.show()
