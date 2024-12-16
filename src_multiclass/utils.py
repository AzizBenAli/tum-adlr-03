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

import matplotlib.pyplot as plt
import os
import torch

def visualize_predictions(model, dataset, shape_idx, device='cpu', output_dir='../plots'):

    model.eval()
    points = []
    true_sdf = []
    labels = []

    with torch.no_grad():
        for batch in dataset:
            batch_shape_idx = batch['shape_idx']
            mask = (batch_shape_idx == shape_idx)
            if mask.any():
                points.append(batch['point'][mask].to(device))
                true_sdf.append(batch['sdf'][mask].to(device))
                labels.append(batch['label_idx'][mask].to(device))

        if len(points) > 0:
            points = torch.cat(points, dim=0)
            true_sdf = torch.cat(true_sdf, dim=0)
            labels = torch.cat(labels, dim=0)
        else:
            print(f"No data found for shape index {shape_idx}.")
            return

        shape_idx_tensor = torch.tensor([shape_idx] * len(points), device=device)

        predicted_sdf = model(shape_idx_tensor, labels, points)

        plt.figure(figsize=(8, 8))
        plt.scatter(predicted_sdf.cpu(), true_sdf.cpu(), c='blue', alpha=0.5, label='Predicted vs True')

        min_sdf = min(predicted_sdf.min().item(), true_sdf.min().item())
        max_sdf = max(predicted_sdf.max().item(), true_sdf.max().item())
        plt.plot([min_sdf, max_sdf], [min_sdf, max_sdf], 'k--', label="Identity Line (x = y)")

        plt.xlabel("Predicted SDF")
        plt.ylabel("True SDF")
        plt.title(f"Predicted vs True SDF for Shape Index {shape_idx}")
        plt.legend()
        plt.grid(True)

        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"shape_{shape_idx}_sdf_plot.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        plt.show()
