import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from PIL import Image


def generate_shape_image(
    model, latent_code, grid_size=224, grid_range=(-10, 10), device="cpu"
):
    grid_size = int(grid_size)
    model.eval()
    with torch.no_grad():
        x = np.linspace(grid_range[0], grid_range[1], grid_size)
        y = np.linspace(grid_range[0], grid_range[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([yy.ravel(), xx.ravel()], axis=-1)
        points = torch.tensor(points, dtype=torch.float32).to(device)

        latent_code_expanded = latent_code.repeat(points.shape[0], 1)

        sdf_values = model.decoder(latent_code_expanded, points)
        sdf_values = sdf_values.cpu().numpy().reshape((grid_size, grid_size))

        image = np.zeros((grid_size, grid_size), dtype=np.uint8)
        image[sdf_values < 0] = 255

    return image


def create_interpolation_animation(
    model,
    interpolated_latents,
    grid_size=224,
    grid_range=(-10, 10),
    device="cpu",
    save_path=None,
):
    images = []
    for z in interpolated_latents:
        image = generate_shape_image(model, z, grid_size, grid_range, device)
        images.append(image)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    img_display = ax.imshow(images[0], cmap="gray", animated=True)

    def update(frame):
        img_display.set_array(images[frame])
        return [img_display]

    anim = FuncAnimation(fig, update, frames=len(images), interval=200, blit=True)

    if save_path:
        anim.save(save_path, writer="pillow", fps=5)

    plt.close(fig)
    return anim


def plot_border_points_on_mask(
    border_points, image_array, output_directory, image_name
):
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

    annotated_image_path = os.path.join(
        annotated_images_dir, f"{image_name}_annotated.png"
    )
    annotated_image.save(annotated_image_path)

    print(f"Image saved to: {annotated_image_path}")
