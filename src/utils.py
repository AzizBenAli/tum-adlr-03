import os
import cv2
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import random_select
import re

def load_images(data_folder):
    """
    Loads images from the specified folder that match a specific angle pattern and are of certain image types.
    """
    angle_pattern = re.compile(r'.*[_\-]20[_\-].*|.*[_\-]20\..*')

    all_files = os.listdir(data_folder)

    matched_files = []
    for f in all_files:
        if angle_pattern.match(f) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            matched_files.append(f)
        else:
            print(f"Not matched: {f}")
            pass

    filenames = sorted(matched_files)

    images = []
    for filename in filenames:
        image_path = os.path.join(data_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is not None:
            images.append((filename, image))
        else:
            print(f"Failed to load image: {filename}")

    print(f"Loaded {len(images)} images at angle '20'.")
    return images

def binarize_images(images):
    """
    Traces the object's outer border, highlights it, fills within the borders with white,
    and sets the background to black.
    """
    processed_images = []
    for filename, image in images:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=10, threshold2=20)
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(grayscale_image)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=2)
            cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)
        else:
            print(f"No contours found in image: {filename}")

        processed_images.append((filename, mask))

    return processed_images


def compute_sdf(images):
    """
    Computes the Signed Distance Function (SDF) for each binary mask.
    """
    sdf_maps = {}
    for filename, mask in images:
        binary_norm = mask / 255.0

        dist_transform_inside = cv2.distanceTransform((binary_norm).astype(np.uint8), cv2.DIST_L2, 5)
        dist_transform_outside = cv2.distanceTransform((1 - binary_norm).astype(np.uint8), cv2.DIST_L2, 5)

        sdf = dist_transform_outside - dist_transform_inside

        sdf_maps[filename] = sdf

    print("Computed Signed Distance Function (SDF) for each binary mask.")
    return sdf_maps

def select_points(sdf_maps, num_total_points=100, border_percentage=0.7):
    """
    Selects points near the object's border as well as some from inside and outside.
    """
    points_info = {}
    for filename, sdf in sdf_maps.items():
        num_border = int(num_total_points * border_percentage)
        num_inside = (num_total_points - num_border) // 2
        num_outside = num_total_points - num_border - num_inside

        threshold_border = 5
        border_indices = np.argwhere(np.abs(sdf) <= threshold_border)
        inside_indices = np.argwhere(sdf <= -threshold_border)
        outside_indices = np.argwhere(sdf >= threshold_border)

        selected_border = random_select(border_indices, num_border)
        selected_inside = random_select(inside_indices, num_inside)
        selected_outside = random_select(outside_indices, num_outside)

        selected_points = np.vstack([
            selected_border, selected_inside, selected_outside
        ]) if selected_border.size else (
            np.vstack([selected_inside, selected_outside]) if selected_inside.size else selected_outside
        )

        selected_sdf_values = sdf[selected_points[:, 0], selected_points[:, 1]]
        point_labels = ['inside' if val < 0 else 'outside' for val in selected_sdf_values]

        points_info[filename] = {
            'points': selected_points,
            'sdf_values': selected_sdf_values,
            'labels': point_labels
        }

    return points_info

def save_points_to_csv(points_info, output_folder):
    """
    Saves the selected points and their SDF values to CSV files.
    """
    csv_folder = os.path.join(output_folder, 'csv_files')
    os.makedirs(csv_folder, exist_ok=True)

    for filename, info in points_info.items():
        points = info['points']
        sdf_values = info['sdf_values']
        labels = info['labels']

        csv_filename = os.path.splitext(filename)[0] + '.csv'
        csv_path = os.path.join(csv_folder, csv_filename)

        with open(csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            csv_writer.writerow(['x', 'y', 'sdf', 'location'])

            for (row, col), sdf_value, label in zip(points, sdf_values, labels):
                csv_writer.writerow([col, row, sdf_value, label])

    print(f"Saved points and SDF values to CSV files in {csv_folder}.")

def annotate_and_save_images(images, points_info, output_folder):
    """
    Annotates all selected points on the processed images.
    Points inside and outside are distinguished by color.
    """
    annotated_folder = os.path.join(output_folder, 'annotated_images')
    os.makedirs(annotated_folder, exist_ok=True)

    for filename, mask in images:
        annotated_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if filename in points_info:
            points = points_info[filename]['points']
            labels = points_info[filename]['labels']
            for (row, col), label in zip(points, labels):
                color = (0, 255, 0) if label == 'inside' else (255, 0, 0)
                cv2.circle(annotated_image, (col, row), radius=3, color=color, thickness=-1)
        output_path = os.path.join(annotated_folder, f"annotated_{filename}")
        cv2.imwrite(output_path, annotated_image)

    print(f"Annotated images saved in {annotated_folder}.")


def plot_random_annotated_images(images, data_folder, output_folder, num_images=20):
    """
    Plots 20 random pairs of original and annotated images, saving the plot to a file.
    """
    plot_dir = os.path.join(".", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    rows, cols = 4, 5
    selected_images = random.sample(images, min(num_images, len(images)))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.02, hspace=0.2)

    for idx, (filename, mask) in enumerate(selected_images):
        orig_path = os.path.join(data_folder, filename)
        original_image = cv2.imread(orig_path, cv2.IMREAD_COLOR)
        annotated_path = os.path.join(output_folder, 'annotated_images', f"annotated_{filename}")
        annotated_image = cv2.imread(annotated_path, cv2.IMREAD_COLOR)

        row, col = idx // cols, idx % cols
        axes[row, col * 2].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[row, col * 2].axis('off')
        axes[row, col * 2 + 1].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        axes[row, col * 2 + 1].axis('off')

    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, "annotated_images.png"))
    plt.show()