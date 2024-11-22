import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import matplotlib
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch


class DataPreprocessor:
    def __init__(self, images, labels, label_names=None):
        """
        Initialize the DataPreprocessor class.

        Parameters:
        - images (numpy.ndarray): Array of BGR images (shape: num_images x 28 x 28 x 3).
        - labels (numpy.ndarray): Array of labels corresponding to the images.
        - label_names (list): List of label names for visualization (optional).
        """
        self.images = images  # BGR images
        self.labels = labels  # Corresponding labels
        self.label_names = label_names if label_names else [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

    def binarize_image_custom(self, filename, image):
        """
        Traces the object's outer border, highlights it, fills within the borders with white,
        and sets the background to black using enhanced edge detection techniques.
        """
        # Step 1: Upscale the image to 112x112 for better edge detection
        upscale_factor = 4
        upscaled_image = cv2.resize(image, (image.shape[1] * upscale_factor, image.shape[0] * upscale_factor),
                                    interpolation=cv2.INTER_LANCZOS4)

        # Step 2: Convert to grayscale
        grayscale_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)

        # Step 3: Apply Bilateral Filter to preserve edges while smoothing
        filtered = cv2.bilateralFilter(grayscale_image, d=9, sigmaColor=75, sigmaSpace=75)

        # Step 4: Apply Unsharp Mask for edge sharpening
        gaussian = cv2.GaussianBlur(filtered, (0, 0), 3)
        unsharp_image = cv2.addWeighted(filtered, 1.5, gaussian, -0.5, 0)

        # Step 5: Compute median for adaptive Canny thresholds
        v = np.median(unsharp_image)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))

        # Step 6: Apply Canny Edge Detection with adaptive thresholds
        edges = cv2.Canny(unsharp_image, threshold1=lower, threshold2=upper)

        # Step 7: Dilate edges to make them more pronounced
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Step 8: Find contours from the dilated edges
        contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 9: Create a blank mask
        mask = np.zeros_like(grayscale_image)

        if contours:
            # Select the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)

            # Draw the contour and fill it
            cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=2)
            cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)
        else:
            print(f"No contours found in image: {filename}")

        # Step 10: Downscale the mask back to original size
        downscaled_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

        return downscaled_mask

    def compute_sdf(self, mask):
        """
        Computes the Signed Distance Function (SDF) for a binary mask.

        Parameters:
        - mask (numpy.ndarray): Binary mask.

        Returns:
        - sdf (numpy.ndarray): Signed Distance Function map.
        """
        dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        sdf = dist_outside - dist_inside
        return sdf

    def select_points(self, sdf, num_total_points=100, border_percentage=0.7):
        """
        Selects points near the object's border as well as some from inside and outside.

        Parameters:
        - sdf (numpy.ndarray): Signed Distance Function map.
        - num_total_points (int): Total number of points to select.
        - border_percentage (float): Percentage of points to select near the border.

        Returns:
        - selected_points (list of dict): List containing points information.
        """
        points = []
        num_border = int(num_total_points * border_percentage)
        num_inside = int((num_total_points - num_border) / 2)
        num_outside = num_total_points - num_border - num_inside

        threshold_border = 5  # Adjust based on image size and requirements
        threshold_inside = threshold_border
        threshold_outside = threshold_border

        # Find indices for each category
        border_indices = np.argwhere(np.abs(sdf) <= threshold_border)
        inside_indices = np.argwhere(sdf <= -threshold_inside)
        outside_indices = np.argwhere(sdf >= threshold_outside)

        # Randomly select points from each category
        selected_border = self._random_select(border_indices, num_border)
        selected_inside = self._random_select(inside_indices, num_inside)
        selected_outside = self._random_select(outside_indices, num_outside)

        # Combine all selected points
        selected_points = []
        if selected_border.size > 0:
            selected_points.extend(selected_border.tolist())
        if selected_inside.size > 0:
            selected_points.extend(selected_inside.tolist())
        if selected_outside.size > 0:
            selected_points.extend(selected_outside.tolist())

        # Get SDF values and labels
        for y, x in selected_points:
            sdf_val = sdf[y, x]
            location = 'inside' if sdf_val < 0 else 'outside'
            points.append({
                'x': x,
                'y': y,
                'sdf': sdf_val,
                'location': location
            })

        return points

    def _random_select(self, indices, num_select):
        """
        Helper function to randomly select a specified number of points from given indices.

        Parameters:
        - indices (numpy.ndarray): Array of point indices.
        - num_select (int): Number of points to select.

        Returns:
        - selected (numpy.ndarray): Selected point indices.
        """
        if indices.size == 0:
            return np.empty((0, 2), dtype=int)
        num_select = min(num_select, indices.shape[0])
        selected = indices[np.random.choice(indices.shape[0], num_select, replace=False)]
        return selected

    def annotate_image(self, mask, points):
        """
        Annotates selected points on the binary mask.

        Parameters:
        - mask (numpy.ndarray): Binary mask.
        - points (list of dict): List containing points information.

        Returns:
        - annotated_image (numpy.ndarray): Annotated image.
        """
        annotated_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        for point in points:
            x, y = point['x'], point['y']
            location = point['location']
            if location == 'inside':
                color = (0, 255, 0)  # Green for inside
            else:
                color = (255, 0, 0)  # Blue for outside
            cv2.circle(annotated_image, (x, y), radius=1, color=color, thickness=-1)

        return annotated_image


def create_directory(path):
    """
    Creates a directory if it does not exist.

    Parameters:
    - path (str): Path of the directory.
    """
    os.makedirs(path, exist_ok=True)


def save_image(image, path):
    """
    Saves an image to the specified path.

    Parameters:
    - image (numpy.ndarray): Image to save.
    - path (str): Path where the image will be saved.
    """
    cv2.imwrite(path, image)


def save_csv(data, path):
    """
    Saves data to a CSV file.

    Parameters:
    - data (list of dict): Data to save.
    - path (str): Path where the CSV will be saved.
    """
    with open(path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(['label', 'x', 'y', 'sdf'])
        # Write data rows
        for entry in data:
            csv_writer.writerow([entry['label'], entry['x'], entry['y'], entry['sdf']])


def plot_samples(original_images, annotated_images, labels, num_samples=10):
    """
    Plots sample original and annotated image pairs.

    Parameters:
    - original_images (list of numpy.ndarray): List of original images.
    - annotated_images (list of numpy.ndarray): List of annotated images.
    - labels (list of str): List of labels corresponding to the images.
    - num_samples (int): Number of samples to plot.
    """
    num_samples = min(num_samples, len(original_images))
    selected_indices = random.sample(range(len(original_images)), num_samples)

    rows = num_samples
    cols = 2  # Original and Annotated

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 4 * rows))
    plt.subplots_adjust(wspace=0.02, hspace=0.2)

    for idx, image_idx in enumerate(selected_indices):
        original_image = original_images[image_idx]
        annotated_image = annotated_images[image_idx]
        label = labels[image_idx]

        # Plot original image
        if num_samples > 1:
            ax_orig = axes[idx, 0]
            ax_annot = axes[idx, 1]
        else:
            ax_orig = axes[0]
            ax_annot = axes[1]

        ax_orig.imshow(original_image, cmap='gray')
        ax_orig.axis('off')
        ax_orig.set_title(f'Original - {label}')

        ax_annot.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        ax_annot.axis('off')
        ax_annot.set_title(f'Annotated - {label}')

    # Maximize the figure window before showing
    try:
        manager = plt.get_current_fig_manager()
        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            manager.window.state('zoomed')  # Maximize window on Windows
        elif backend in ['Qt4Agg', 'Qt5Agg']:
            manager.window.showMaximized()  # Maximize window on Qt backends
        elif backend == 'WXAgg':
            manager.window.Maximize()
        elif 'GTK' in backend:
            manager.window.maximize()
        elif 'macosx' in backend:
            manager.window.set_fullscreen(True)
        else:
            print(f"Backend '{backend}' does not support automatic full-screen toggling.")
            # As a fallback, toggle full screen if possible
            if hasattr(manager.window, 'showFullScreen'):
                manager.window.showFullScreen(True)
            elif hasattr(manager.window, 'full_screen_toggle'):
                manager.window.full_screen_toggle()
    except Exception as e:
        print(f"Could not maximize the window: {e}")
        # As a fallback, use full_screen_toggle
        try:
            fig.canvas.manager.full_screen_toggle()
        except AttributeError:
            print("Full screen toggle not supported for this backend.")

    plt.tight_layout()
    plt.show()


def main():
    # Define label names (replace spaces with underscores for directory names)
    label_names = [
        "T-shirt_top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"
    ]

    # Step 1: Load the Fashion-MNIST dataset using Torchvision
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Download and load the training data
    train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

    # Combine train and test datasets
    all_images = torch.cat((train_dataset.data, test_dataset.data), dim=0).numpy()
    all_labels = np.concatenate((train_dataset.targets.numpy(), test_dataset.targets.numpy()), axis=0)

    print(f"Total images loaded: {all_images.shape[0]}")

    # Step 2: Initialize the DataPreprocessor
    # Convert grayscale images to BGR format for cv2 compatibility
    # Since binarization function expects BGR images
    all_images_bgr = []
    for img in all_images:
        # Convert single channel grayscale to BGR by duplicating channels
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        all_images_bgr.append(img_bgr)
    all_images_bgr = np.array(all_images_bgr)

    preprocessor = DataPreprocessor(images=all_images_bgr, labels=all_labels, label_names=label_names)

    # Step 3: Create the parent output directory
    parent_output_dir = "../output_mnist"
    create_directory(parent_output_dir)

    # Step 4: Process each label
    for label_idx, label_name in enumerate(label_names):
        readable_label = label_name.replace('_', ' ')
        print(f"\nProcessing label: {readable_label}")

        # Create label-specific directory
        label_dir = os.path.join(parent_output_dir, label_name)
        create_directory(label_dir)

        # Create child directories
        original_dir = os.path.join(label_dir, "original_images")
        annotated_dir = os.path.join(label_dir, "annotated_images")
        csv_dir = os.path.join(label_dir, "csv_files")

        create_directory(original_dir)
        create_directory(annotated_dir)
        create_directory(csv_dir)

        # Find all image indices with the current label
        label_image_indices = np.where(all_labels == label_idx)[0]
        print(f"Total images for label '{readable_label}': {len(label_image_indices)}")

        # Ensure there are at least 100 images
        if len(label_image_indices) < 100:
            print(
                f"Not enough images for label '{readable_label}'. Required: 100, Available: {len(label_image_indices)}")
            continue

        # Randomly select 100 image indices
        selected_indices = np.random.choice(label_image_indices, 100, replace=False)

        for img_num, img_idx in enumerate(selected_indices, 1):
            # Original image in BGR
            original_image = preprocessor.images[img_idx]

            # Assign filename with zero-padded numbering
            img_num_padded = f"{img_num:03d}"
            original_filename = f"original_{img_num_padded}.png"

            # Save original image
            original_path = os.path.join(original_dir, original_filename)
            save_image(original_image, original_path)

            # Binarize image using custom method
            binarized_mask = preprocessor.binarize_image_custom(original_filename, original_image)

            # Save binarized mask as annotated image before adding points
            annotated_filename = f"annotated_{img_num_padded}.png"
            annotated_path = os.path.join(annotated_dir, annotated_filename)
            # Initially save the binarized mask
            save_image(binarized_mask, annotated_path)

            # Compute SDF
            sdf = preprocessor.compute_sdf(binarized_mask)

            # Select points
            points = preprocessor.select_points(sdf, num_total_points=100, border_percentage=0.7)

            # Annotate image with selected points
            annotated_image = preprocessor.annotate_image(binarized_mask, points)

            # Save annotated image
            save_image(annotated_image, annotated_path)

            # Prepare CSV entry
            csv_data = []
            for point in points:
                csv_data.append({
                    'label': readable_label,
                    'x': point['x'],
                    'y': point['y'],
                    'sdf': point['sdf']
                })

            # Save CSV file
            csv_filename = f"data_{img_num_padded}.csv"
            csv_path = os.path.join(csv_dir, csv_filename)
            save_csv(csv_data, csv_path)

            # Progress update every 10 images
            if img_num % 10 == 0:
                print(f"  Processed {img_num}/100 images for label '{readable_label}'.")

    # Step 5: Plot sample images
    # For visualization, collect some original and annotated images across different labels
    sample_originals = []
    sample_annotated = []
    sample_labels = []

    for label_idx, label_name in enumerate(label_names):
        readable_label = label_name.replace('_', ' ')
        label_dir = os.path.join(parent_output_dir, label_name)
        if not os.path.exists(label_dir):
            continue
        original_dir = os.path.join(label_dir, "original_images")
        annotated_dir = os.path.join(label_dir, "annotated_images")
        # Select first 10 images for sampling
        for img_num in range(1, 11):
            img_num_padded = f"{img_num:03d}"
            original_path = os.path.join(original_dir, f"original_{img_num_padded}.png")
            annotated_path = os.path.join(annotated_dir, f"annotated_{img_num_padded}.png")
            if os.path.exists(original_path) and os.path.exists(annotated_path):
                original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
                annotated_image = cv2.imread(annotated_path, cv2.IMREAD_COLOR)
                sample_originals.append(original_image)
                sample_annotated.append(annotated_image)
                sample_labels.append(readable_label)
            if len(sample_originals) >= 10:
                break
        if len(sample_originals) >= 10:
            break

    if sample_originals and sample_annotated:
        plot_samples(sample_originals, sample_annotated, sample_labels, num_samples=10)  # Adjust num_samples as needed
    else:
        print("No samples available for plotting.")

    # Step 6: Confirmation
    print("\nProcessing complete. All outputs saved in 'output_mnist' directory.")


if __name__ == "__main__":
    main()